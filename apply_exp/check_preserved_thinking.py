#!/usr/bin/env python3
"""Check Qwen3.8 thinking across live Chat Completions requests (requires Jinja2)."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jinja2 import TemplateError, nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


TOOLS = [{
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two integers and return their product.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}]
THINK_BLOCK = re.compile(r"<\|im_start\|>assistant\n<think>\n(.*?)\n</think>\n\n", re.S)


class GenerationBlock(Extension):
    """Render HF's generation block as text without collecting a token mask."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.Scope(body).set_lineno(lineno)


def raise_template_error(message):
    raise TemplateError(message)


def load_template(path):
    if path.is_dir():
        path = path / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[GenerationBlock, "jinja2.ext.loopcontrols"]
    )
    env.globals["raise_exception"] = raise_template_error
    # Match the JSON serialization used by Hugging Face's chat renderer.
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    return env.from_string(path.read_text(encoding="utf-8"))


def apply_template(template, messages, **kwargs):
    history = deepcopy(messages)
    for message in history:
        for call in message.get("tool_calls", []):
            # The HTTP API uses a JSON string, while Qwen's Jinja iterates a mapping.
            function = call["function"]
            function["arguments"] = json.loads(function["arguments"])
    return template.render(
        messages=history, tools=TOOLS, add_generation_prompt=True, enable_thinking=True, **kwargs
    )


def check_prompt(label, prompt, expected, show_prompt=False):
    if show_prompt:
        print(f"\n--- {label} prompt ---\n{prompt}--- end prompt ---", flush=True)
    actual = THINK_BLOCK.findall(prompt)
    if actual != expected:
        raise RuntimeError(
            f"{label}: expected {len(expected)} historical thinking blocks with matching text, "
            f"found {len(actual)}. Re-run with --show-prompt to inspect the prompt."
        )
    print(f"{label}: {len(actual)} historical thinking block(s) match exactly", flush=True)


def complete(args, url, messages):
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key := os.getenv("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urlopen(request, timeout=args.timeout) as response:
            choice = json.load(response)["choices"][0]
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}") from error
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Completion hit max_tokens. Increase --max-tokens and retry.")
    reply = choice["message"]
    reasoning = reply.get("reasoning_content") or reply.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError("No reasoning in the API response. Check the server's reasoning parser and thinking mode.")
    message = {"role": "assistant", "content": reply.get("content") or "", "reasoning_content": reasoning}
    if reply.get("tool_calls"):
        message["tool_calls"] = reply["tool_calls"]
    return message


def multiply_result(message):
    calls = message.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "multiply":
        raise RuntimeError("Expected one multiply tool call. Check the server's automatic tool-call parser.")
    call = calls[0]
    arguments = json.loads(call["function"]["arguments"])
    if arguments not in ({"a": 17, "b": 23}, {"a": 23, "b": 17}):
        raise RuntimeError(f"Unexpected multiply arguments: {arguments!r}")
    result = arguments["a"] * arguments["b"]
    return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"product": result})}


def run(args):
    template = load_template(args.template_path)
    host = f"[{args.ip}]" if ":" in args.ip and not args.ip.startswith("[") else args.ip
    url = f"http://{host}:{args.port}/v1/chat/completions"
    messages = [
        {"role": "system", "content": "Use multiply when requested. After receiving its result, answer briefly."},
        {"role": "user", "content": "Use the multiply tool once to compute 17 times 23, then report the result."},
    ]
    reasoning_history = []
    stages = ("request 1: initial question", "request 2: tool result", "request 3: new user turn")
    for index, label in enumerate(stages):
        prompt = apply_template(template, messages)
        check_prompt(label, prompt, reasoning_history, args.show_prompt)
        if index in (1, 2):
            control = apply_template(template, messages, preserve_thinking=False)
            expected = reasoning_history if index == 1 else []
            check_prompt(f"{label} (local preserve_thinking=false control)", control, expected, args.show_prompt)

        message = complete(args, url, messages)
        messages.append(message)
        reasoning_history.append(message["reasoning_content"].strip())
        if index == 0:
            result = multiply_result(message)
            messages.append(result)
            print(f"Executed multiply locally and appended tool result {result['content']}", flush=True)
        else:
            if message.get("tool_calls") or not message["content"].strip():
                raise RuntimeError(f"{label}: expected a final text response")
            print(f"Assistant: {message['content']}", flush=True)
            if index == 1:
                messages.append({"role": "user", "content": "Add 7 to that result without calling a tool."})

    check_prompt("final local render", apply_template(template, messages), reasoning_history, args.show_prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True, help="Server IP or hostname")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model", "--model-name", required=True, help="Model name exposed by the server")
    parser.add_argument(
        "--template-path", "--model-path", required=True, type=Path,
        help="Local chat_template.jinja file or model directory containing it",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180, help="HTTP timeout in seconds")
    parser.add_argument("--show-prompt", action="store_true", help="Print the full locally rendered prompt")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, URLError, TemplateError, RuntimeError, ValueError, KeyError, IndexError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
