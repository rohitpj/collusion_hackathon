"""Inspect provider for the pinned Dolos/Qwen weights; no GPU imports at module load."""

import json
import re
from uuid import uuid4

from inspect_ai.model import (
    ChatCompletionChoice, ChatMessageAssistant, GenerateConfig, ModelAPI,
    ModelCall, ModelOutput, ModelUsage, modelapi,
)
from inspect_ai.tool import ToolCall


def chat_messages(messages):
    """Translate Inspect messages to Qwen's native template, including tool errors."""
    rows = []
    for message in messages:
        row = {"role": message.role, "content": message.text}
        if message.role == "assistant" and message.tool_calls:
            row["tool_calls"] = [
                {"type": "function", "function": {"name": c.function, "arguments": c.arguments}}
                for c in message.tool_calls
            ]
        elif message.role == "tool":
            row["name"] = message.function or ""
            if message.error:
                row["content"] = f"Tool error ({message.error.type}): {message.error.message}"
        rows.append(row)
    return rows


def parse_response(raw, parser="strict"):
    """Preserve duplicates and expose malformed calls to Inspect as parsing errors.

    strict requires the native closing tag. json accepts a complete JSON object even
    if its closing tag is damaged; every such recovery is recorded in diagnostics.
    """
    if parser not in ("strict", "json"):
        raise ValueError("tool parser must be strict or json")
    calls, diagnostics, spans = [], [], []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"<tool_call>\s*", raw):
        if spans and match.start() < spans[-1][1]:
            continue  # A literal marker inside JSON arguments is not another call.
        end = match.end()
        function, arguments, error = "__invalid_tool_call__", {}, None
        try:
            data, consumed = decoder.raw_decode(raw[end:])
            end += consumed
            if not isinstance(data, dict) or not isinstance(data.get("name"), str):
                raise ValueError("tool call must be an object with a string name")
            function = data["name"]
            if not isinstance(data.get("arguments"), dict):
                raise ValueError("arguments must be a JSON object")
            arguments = data["arguments"]
            close = re.match(r"\s*</tool_call>", raw[end:])
            if close:
                end += close.end()
            else:
                diagnostics.append({"offset": match.start(), "issue": "missing_closing_tag"})
                if parser == "strict":
                    raise ValueError("missing </tool_call>")
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            diagnostics.append({"offset": match.start(), "issue": "parse_error", "error": error})
            next_open = raw.find("<tool_call>", end)
            end = next_open if next_open >= 0 else len(raw)
        calls.append(ToolCall(id=uuid4().hex, function=function, arguments=arguments, parse_error=error))
        spans.append((match.start(), end))
    # Keep prose separate so a pure tool call is not misclassified as refusal text.
    content, cursor = [], 0
    for start, end in spans:
        content.append(raw[cursor:start])
        cursor = end
    content.append(raw[cursor:])
    return "".join(content).strip(), calls, diagnostics


class DolosHF(ModelAPI):
    def __init__(self, model_name, base_url=None, api_key=None,
                 config=GenerateConfig(), tool_parser="strict", **kwargs):
        if model_name not in ("dolos", "qwen"):
            raise ValueError("model must be dolos or qwen")
        if kwargs:
            raise ValueError(f"unsupported model arguments: {sorted(kwargs)}")
        if tool_parser not in ("strict", "json"):
            raise ValueError("tool_parser must be strict or json")
        super().__init__(model_name, base_url, api_key, [], config)
        self.tool_parser = tool_parser
        self.model = self.tokenizer = None
        self._seed = None

    def max_connections(self):
        return 1

    def _load(self):
        if self.model is None:
            import torch
            from transformers import AutoTokenizer
            import generate as g

            self.tokenizer = AutoTokenizer.from_pretrained(g.QWEN_REPO, revision=g.QWEN_REVISION, token=False)
            self.model = g.load_model(self.model_name, torch.device("cuda"), torch.bfloat16)

    async def generate(self, input, tools, tool_choice, config):
        if tool_choice not in ("auto", "none"):
            raise ValueError("DolosHF supports auto/none tool choice; forced calls are not implemented")
        import torch
        from transformers import set_seed

        self._load()
        messages = chat_messages(input)
        schemas = [{"type": "function", "function": t.model_dump(exclude_none=True)} for t in tools]
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=schemas if schemas and tool_choice != "none" else None,
            tokenize=False, add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        n_input = encoded.input_ids.shape[1]
        n_output = config.max_tokens or 4096
        if n_input + n_output > self.model.config.max_position_embeddings:
            raise ValueError("context plus requested output exceeds model context window; no truncation applied")
        temperature = config.temperature if config.temperature is not None else 0.0
        seed = config.seed if config.seed is not None else 0
        if self._seed != seed:
            set_seed(seed)
            self._seed = seed
        eos = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        options = dict(max_new_tokens=n_output, do_sample=temperature > 0,
                       eos_token_id=eos, pad_token_id=eos[-1])
        if temperature > 0:
            options.update(temperature=temperature, top_p=config.top_p if config.top_p is not None else 1.0,
                           top_k=config.top_k if config.top_k is not None else 0)
        with torch.inference_mode():
            output = self.model.generate(**encoded, **options)[0, n_input:]
        raw = self.tokenizer.decode(output, skip_special_tokens=False)
        text = raw
        for token in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(token)[0]
        content, calls, diagnostics = parse_response(text, self.tool_parser)
        # Native tool-choice=none means tools are not offered or dispatched.
        if tool_choice == "none":
            content, calls = text, []
        truncated = len(output) >= n_output and int(output[-1]) not in eos
        stop = "max_tokens" if truncated else "tool_calls" if calls else "stop"
        result = ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=content, tool_calls=calls or None),
                                          stop_reason=stop)],
            usage=ModelUsage(input_tokens=n_input, output_tokens=len(output), total_tokens=n_input + len(output)),
            metadata={"raw_response": raw, "tool_parser": self.tool_parser, "parse_diagnostics": diagnostics},
        )
        return result, ModelCall(request={"prompt": prompt, "generation": options}, response={"text": raw})


@modelapi(name="dolos_hf")
def dolos_hf():
    return DolosHF
