#!/usr/bin/env python3
"""Recover structured training records from Qwen3.8 rendered messages."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


# These strings identify the generated system prelude in Qwen3.8's chat template.
REASONING_PREFIXES = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.",
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.",
)
TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
TOOL_RULES_START = (
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:"
)
CALL_START = re.compile(r"^[ \t]*<tool_call>[ \t]*\r?\n", re.MULTILINE)
FUNCTION_START = re.compile(r"<function=([^\s>]+)>\s*")
PARAMETER = re.compile(r"<parameter=([^\s>]+)>\r?\n(.*?)\r?\n</parameter>", re.DOTALL)
CALL_END = re.compile(r"</function>\s*</tool_call>")
RESPONSE = re.compile(r"<tool_response>\r?\n(.*?)\r?\n</tool_response>", re.DOTALL)
RESPONSE_SEPARATOR = re.compile(r"---\s+(?=<tool_response>)")
TARGET_METADATA = ("session_id", "category", "split_type", "split_index", "status")


def split_system(content):
    for prefix in REASONING_PREFIXES:
        if content == prefix:
            return "", []
        if content.startswith(prefix + "\n\n"):
            content = content[len(prefix) + 2:]
            break

    if not content.startswith(TOOLS_HEADER):
        return content, []

    definitions, closing, suffix = content[len(TOOLS_HEADER):].partition("\n</tools>")
    if not closing or not suffix.startswith(TOOL_RULES_START):
        raise ValueError("invalid Qwen system tool prelude")
    _, end_rules, system = suffix.partition("\n</IMPORTANT>")
    if not end_rules or (system and not system.startswith("\n\n")):
        raise ValueError("missing end of Qwen system tool rules")
    tools = [json.loads(line) for line in definitions.split("\n")]
    return system[2:], tools


def skip_whitespace(text, position):
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def decode_argument(text, schema):
    # The template prints strings verbatim and JSON-encodes other values.
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if "string" in types or declared is None:
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    json_type = {
        dict: "object", list: "array", bool: "boolean", type(None): "null",
        int: "integer", float: "number",
    }.get(type(value))
    if json_type in types or (json_type == "integer" and "number" in types):
        return value
    return text


def split_calls(content, schemas):
    calls, text = [], []
    position = 0
    while start := CALL_START.search(content, position):
        text.append(content[position:start.start()])
        function = FUNCTION_START.match(content, skip_whitespace(content, start.end()))
        if not function:
            raise ValueError("missing function name in tool call")
        name = function[1]
        arguments = {}
        position = function.end()
        properties = schemas.get(name, {})
        while parameter := PARAMETER.match(content, position):
            key, value = parameter.groups()
            if key in arguments:
                raise ValueError(f"duplicate parameter {key!r} in {name!r}")
            arguments[key] = decode_argument(value, properties.get(key, {}))
            position = skip_whitespace(content, parameter.end())
        end = CALL_END.match(content, position)
        if not end:
            raise ValueError(f"malformed tool call to {name!r}")
        calls.append({"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)})
        position = end.end()

    text.append(content[position:])
    return "".join(text).strip(), calls


def convert_record(record, index):
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if record.get("images"):
        raise ValueError("multimodal records are unsupported")

    conversations, tools = [], []
    schemas = {}
    pending_calls = deque()
    call_index = 0

    for number, message in enumerate(messages, 1):
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError(f"message {number}: content must be text")
        if role == "system" and number == 1:
            content, tools = split_system(content)
            schemas = {
                tool["function"]["name"]: tool["function"].get("parameters", {}).get(
                    "properties", {}
                )
                for tool in tools
            }
        elif role == "user" and content.lstrip().startswith("<tool_response>"):
            position = skip_whitespace(content, 0)
            while position < len(content):
                response = RESPONSE.match(content, position)
                if not response or not pending_calls:
                    raise ValueError(f"message {number}: malformed or unmatched tool response")
                conversations.append({
                    "role": "tool",
                    "content": response[1],
                    "tool_call_id": pending_calls.popleft(),
                })
                position = skip_whitespace(content, response.end())
                # The source export sometimes inserts a divider between results.
                divider = RESPONSE_SEPARATOR.match(content, position)
                if divider:
                    position = divider.end()
            continue
        elif role not in ("assistant", "user"):
            raise ValueError(f"message {number}: unexpected role {role!r}")

        pending_calls.clear()
        turn = {"role": role, "content": content}
        if role == "assistant":
            content = content.lstrip()
            if content.startswith("<think>"):
                reasoning, _, content = content[len("<think>"):].partition("</think>")
                turn["reasoning_content"] = reasoning.strip()

            turn["content"], calls = split_calls(content, schemas)
            if calls:
                turn["tool_calls"] = []
                for function in calls:
                    call_index += 1
                    call_id = f"call_{index:05d}_{call_index}"
                    pending_calls.append(call_id)
                    turn["tool_calls"].append({
                        "id": call_id,
                        "type": "function",
                        "function": function,
                    })
        conversations.append(turn)

    result = {"id": f"distill_data_{index:05d}"}
    result.update({key: record[key] for key in TARGET_METADATA if key in record})
    result.update(conversations=conversations, tools=tools)
    return result


def read_records(source):
    """Stream JSONL, with support for the repository's indented single record."""
    index = 0
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        index += 1
        if index == 1 and line.strip() == "{":
            line += source.read()
        yield index, line_number, line


def convert_line(task):
    index, line_number, line = task
    try:
        return json.dumps(convert_record(json.loads(line), index), ensure_ascii=False)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"input line {line_number} (record {index}): {exc}") from exc


def write_records(source, output, workers):
    records = read_records(source)
    if workers == 1:
        for line in map(convert_line, records):
            output.write(line + "\n")
        return

    # Bound queued work while keeping input order and stable IDs.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while batch := list(islice(records, 32 * workers)):
            for line in pool.map(convert_line, batch, chunksize=4):
                output.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source JSONL or indented JSON object")
    parser.add_argument("output", type=Path, help="destination JSONL")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    temporary = None
    try:
        with args.input.open(encoding="utf-8") as source, NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            write_records(source, output, args.workers)
        temporary.replace(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
