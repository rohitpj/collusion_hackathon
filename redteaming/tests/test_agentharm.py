"""Offline contract tests; run with python -m unittest discover -s tests -v."""

import tempfile
import unittest
import json
import os
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool,
    ChatMessageUser, ModelOutput, get_model,
)
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolCall, ToolCallError, tool

from agentharm_adapter import chat_messages, parse_response
from agentharm_modal import optional_bool, selection, preflight_judge


@tool
def echo_test():
    async def execute(value: str):
        """Return the supplied value.

        Args:
            value: Text to echo.
        """
        return value
    return execute


class AdapterTests(unittest.TestCase):
    def test_missing_judge_key_fails_before_modal(self):
        import agentharm_run
        with patch.dict(os.environ, {}, clear=True), patch("agentharm_run.subprocess.run") as launch:
            self.assertEqual(agentharm_run.main([]), 2)
            launch.assert_not_called()

    def test_placeholder_judge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            preflight_judge(True, "openai-api/JUDGE_MODEL_ID", "OPENROUTER_API_KEY")

    def test_judge_env_file_and_shell_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENAI_API_KEY=file-test-value\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(preflight_judge(True, "openai/judge", "OPENAI_API_KEY", env_file=str(path)),
                                 "file-test-value")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-test-value"}, clear=True):
                self.assertEqual(preflight_judge(True, "openai/judge", "OPENAI_API_KEY", env_file=str(path)),
                                 "shell-test-value")

    def test_no_score_needs_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(preflight_judge(False, "openai/judge", "OPENAI_API_KEY"))

    def test_named_secret_checked_before_launch(self):
        import agentharm_run
        with patch("modal.Secret.from_name") as secret, patch("agentharm_run.subprocess.run") as launch:
            secret.return_value.hydrate.side_effect = ValueError("secret does not exist")
            self.assertEqual(agentharm_run.main(["--secret-name", "missing-secret"]), 2)
            launch.assert_not_called()

    def test_launcher_enables_utf8(self):
        import agentharm_run
        with patch("agentharm_run.subprocess.run") as launch:
            launch.return_value.returncode = 0
            self.assertEqual(agentharm_run.main(["--no-score", "--limit", "1"]), 0)
            self.assertEqual(launch.call_args.kwargs["env"]["PYTHONUTF8"], "1")

    def test_valid_duplicate_calls_and_prose(self):
        block = '<tool_call>{"name":"echo_test","arguments":{"value":"a"}}</tool_call>'
        text, calls, diagnostics = parse_response("Thinking.\n" + block + block)
        self.assertEqual(text, "Thinking.")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0].id, calls[1].id)
        self.assertFalse(diagnostics)
        self.assertFalse(calls[0].parse_error)

    def test_broken_closing_tag_is_error_in_strict_mode(self):
        raw = '<tool_call>{"name":"echo_test","arguments":{"value":"a"}}<tool_call>'
        _, calls, diagnostics = parse_response(raw)
        self.assertTrue(calls[0].parse_error)
        self.assertEqual(diagnostics[0]["issue"], "missing_closing_tag")

    def test_json_recovery_is_explicit(self):
        raw = '<tool_call>{"name":"echo_test","arguments":{"value":"a"}}'
        _, calls, diagnostics = parse_response(raw, "json")
        self.assertFalse(calls[0].parse_error)
        self.assertEqual(diagnostics[0]["issue"], "missing_closing_tag")

    def test_malformed_call_does_not_disappear(self):
        for payload in ('not json', '{"name":"x","arguments":"oops"}', '[1,2]'):
            _, calls, _ = parse_response("<tool_call>" + payload + "</tool_call>")
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].parse_error)

    def test_marker_inside_argument_is_not_dispatched(self):
        payload = json.dumps({"name": "echo_test", "arguments": {"value": "literal <tool_call> marker"}})
        _, calls, _ = parse_response("<tool_call>" + payload + "</tool_call>")
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0].parse_error)

    def test_message_conversion_and_tool_errors(self):
        messages = [ChatMessageSystem(content="system"), ChatMessageUser(content="hello"),
                    ChatMessageAssistant(content="", tool_calls=[
                        ToolCall(id="1", function="echo_test", arguments={"value": "hello"})]),
                    ChatMessageTool(content="", function="echo_test", tool_call_id="1",
                                    error=ToolCallError(type="parsing", message="bad JSON"))]
        rows = chat_messages(messages)
        self.assertEqual(rows[2]["tool_calls"][0]["function"]["arguments"], {"value": "hello"})
        self.assertIn("bad JSON", rows[3]["content"])

    def test_provider_registration_and_invalid_names(self):
        model = get_model("dolos_hf/dolos", tool_parser="json")
        self.assertEqual(model.api.tool_parser, "json")
        with self.assertRaises(ValueError):
            get_model("dolos_hf/other")

    def test_options(self):
        self.assertEqual(selection("both", ("dolos", "qwen"), "model"), ["dolos", "qwen"])
        self.assertIsNone(optional_bool("all"))
        self.assertIs(optional_bool("no"), False)
        with self.assertRaises(ValueError):
            optional_bool("maybe")

    @patch("inspect_ai._util.entrypoints.entry_points", return_value=[])
    def test_inspect_dispatch_and_error_recovery(self, _entrypoints):
        """Real Inspect tool loop: malformed call -> tool error -> valid call -> answer."""
        generations = iter([
            '<tool_call>{"name":"echo_test","arguments":{"value":"hello"}}',
            '<tool_call>{"name":"echo_test","arguments":{"value":"hello"}}</tool_call>',
            'Finished.',
        ])

        def outputs(input, tools, tool_choice, config):
            text, calls, diagnostics = parse_response(next(generations))
            return ModelOutput(model="adapter-test", choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=text, tool_calls=calls or None),
                stop_reason="tool_calls" if calls else "stop")])

        from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer
        module = ModuleType("test_grading")

        async def grade(metadata, messages):
            return {"echo_completed": float(any(m.role == "tool" and m.text == "hello" for m in messages))}

        module.echo_grade = grade
        judge = get_model("mockllm/judge", custom_outputs=lambda *args:
                          ModelOutput.from_content(model="judge", content="Answer: No"))

        # This targeted offline test does not need the registry of unrelated evals.
        with tempfile.TemporaryDirectory() as directory, patch(
            "inspect_ai._util.entrypoints.entry_points", return_value=[]
        ):
            logs = inspect_eval(
                Task(dataset=[Sample(input="Echo hello", metadata={
                         "grading_function": "echo_grade", "category": "test"})],
                     solver=[use_tools(echo_test()), generate()], max_messages=10,
                     scorer=combined_scorer(module, RefusalJudgeLLM(judge), judge)),
                model=get_model("mockllm/adapter-test", custom_outputs=outputs),
                display="none", log_dir=str(Path(directory) / "logs"),
            )
            self.assertEqual(logs[0].status, "success")
            messages = logs[0].samples[0].messages
            results = [m for m in messages if m.role == "tool"]
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].error.type, "parsing")
            self.assertEqual(results[1].text, "hello")
            self.assertEqual(messages[-1].text, "Finished.")
            score = next(iter(logs[0].samples[0].scores.values()))
            self.assertEqual(score.value, {"score": 1.0, "refusal": 0.0})


if __name__ == "__main__":
    unittest.main()
