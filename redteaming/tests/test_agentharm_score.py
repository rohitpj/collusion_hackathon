"""Offline checks for credential loading and replay-only official scoring."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.solver import generate
from inspect_evals.agentharm.benchmark import benign_grading_functions

from agentharm_modal import preflight_judge
from agentharm_score import score_log
from agentharm_score_audit import audited_metrics


class ReplayTests(unittest.TestCase):
    def test_key_file_is_read_without_environment_mutation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "key.txt"
            path.write_text("fixture-value\n", encoding="utf-8")
            self.assertEqual(preflight_judge(True, "openrouter/test", "OPENROUTER_API_KEY",
                                            key_file=str(path)), "fixture-value")
            self.assertNotIn("OPENROUTER_API_KEY", os.environ)

    @patch("inspect_ai._util.entrypoints.entry_points", return_value=[])
    def test_replay_scores_without_changing_original(self, _entrypoints):
        async def fixture_grade(metadata, messages):
            return {"answer_present": float(messages[-1].text == "Saved candidate response")}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            benign_grading_functions, "fixture_grade", fixture_grade, create=True
        ):
            root = Path(directory)
            logs = inspect_eval(
                Task(name="agentharm_benign", dataset=[Sample(input="A fixture", metadata={
                    "grading_function": "fixture_grade", "category": "test"})], solver=generate()),
                model=get_model("mockllm/saved-candidate", custom_outputs=[
                    ModelOutput.from_content(model="saved-candidate", content="Saved candidate response")]),
                score=False, display="none", log_dir=str(root / "initial"))
            source, destination = root / "original.eval", root / "scored.eval"
            write_eval_log(logs[0], source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            judge = get_model("mockllm/replay-test-judge", custom_outputs=lambda *args:
                              ModelOutput.from_content(model="judge", content="Answer: No"))
            summary = score_log(source, destination, judge)
            self.assertEqual(summary["sample_errors"], 0)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            result = read_eval_log(str(destination))
            self.assertEqual(result.samples[0].messages[-1].text, "Saved candidate response")
            self.assertEqual(next(iter(result.samples[0].scores.values())).value,
                             {"score": 1.0, "refusal": 0.0})
            self.assertEqual(audited_metrics(result)["scored_samples"], 1)
            self.assertEqual(audited_metrics(result)["metrics"]["avg_score"], 1.0)
            event = SimpleNamespace(event="model", model="openrouter/test",
                                    input=[ChatMessageUser(content="Classify <response>text</response>")],
                                    output=ModelOutput.from_content(model="judge", content="Answer: No"))
            result.samples[0].events.append(event)
            self.assertEqual(audited_metrics(result)["scored_samples"], 1)
            event.input = [ChatMessageUser(content="Say pass or fail")]
            event.output = ModelOutput.from_content(model="judge", content="Please provide the text")
            self.assertEqual(audited_metrics(result)["scored_samples"], 0)
            result.samples[0].events.pop()
            from inspect_ai.event import InfoEvent
            result.samples[0].events.append(InfoEvent(data="Semantic judge response: `needs more information`"))
            self.assertEqual(audited_metrics(result)["scored_samples"], 0)

            async def broken_grade(metadata, messages):
                raise KeyError("file_path")

            with patch.object(benign_grading_functions, "fixture_grade", broken_grade):
                failure = score_log(source, root / "unscorable.eval", judge)
            self.assertEqual(failure["scored_samples"], 0)
            self.assertIn("file_path", failure["scoring_errors"][0]["error"])
            import math
            unscorable = read_eval_log(str(root / "unscorable.eval"))
            value = next(iter(unscorable.samples[0].scores.values())).value
            self.assertTrue(math.isnan(value["score"]))
            self.assertIsNone(value["refusal"])
            self.assertEqual(audited_metrics(unscorable)["scored_samples"], 0)


if __name__ == "__main__":
    unittest.main()
