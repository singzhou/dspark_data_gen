#!/usr/bin/env python3
"""Recover structured training records from Qwen3.8 rendered messages."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, islice
import json
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile


# These strings identify the generated system prelude in Qwen3.8's chat template.
REASONING_PREFIXES = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.",
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration.",
)
TOOLS_HEADER = re.compile(
    r"#\s*Tools?\s+You have access to the following functions:\s*<tools>\s*"
)
TOOL_RULES_START = "If you choose to call a function ONLY reply in the following format"
CHAT_BOUNDARY = re.compile(r"<\|im_end\|>\s*<\|im_start\|>(user|assistant)\r?\n")
CALL_START = re.compile(r"<tool_call>|^[ \t]*(?=<function[=>])", re.MULTILINE)
FUNCTION_START = re.compile(r"(?:<?function=|<function>|=)([^\s<>]+)>\s*")
PARAMETER_START = re.compile(r"(?:<parameter=|</tool_call>=)([^\s<>]+)>")
PARAMETER_END = "</parameter>"
RESPONSE = re.compile(r"<tool_response>\r?\n(.*?)\r?\n</tool_response>", re.DOTALL)
RESPONSE_SEPARATOR = re.compile(r"---\s+(?=<tool_response>)")
RESPONSE_ID = re.compile(
    r"<system-reminder>\s*TOOL_RESPONSE_ID=([^,\s]+),\s*TOOL_NAME=([^<\s]+)\s*</system-reminder>"
)
TARGET_METADATA = ("session_id", "category", "split_type", "split_index", "status")


def split_system(content):
    for prefix in REASONING_PREFIXES:
        if content == prefix:
            return "", []
        if content.startswith(prefix + "\n\n"):
            content = content[len(prefix) + 2:]
            break

    header = TOOLS_HEADER.match(content)
    if not header:
        return content, []

    definitions, closing, suffix = content[header.end():].partition("</tools>")
    if not closing or not suffix.lstrip().startswith(TOOL_RULES_START):
        raise ValueError("invalid Qwen system tool prelude")
    _, end_rules, system = suffix.partition("</IMPORTANT>")
    if not end_rules:
        raise ValueError("missing end of Qwen system tool rules")

    tools = []
    decoder = json.JSONDecoder()
    position = skip_whitespace(definitions, 0)
    while position < len(definitions):
        tool, end = decoder.raw_decode(definitions, position)
        if "function" not in tool:
            function = dict(tool)
            function["parameters"] = function.pop("arguments")
            tool = {"type": "function", "function": function}
        tools.append(tool)
        position = skip_whitespace(definitions, end)
    return system.lstrip("\r\n"), tools


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


def parse_parameters(text, position, properties):
    arguments = {}
    while parameter := PARAMETER_START.match(text, position):
        key = parameter[1]
        if key in arguments:
            raise ValueError(f"duplicate parameter {key!r}")
        start = parameter.end()
        value_start = skip_whitespace(text, start)
        schema = properties.get(key, {})
        if PARAMETER_START.match(text, value_start):
            value, end = parse_parameters(text, value_start, schema.get("properties", {}))
            if not text.startswith(PARAMETER_END, end):
                raise ValueError(f"missing closing tag for nested parameter {key!r}")
        else:
            end = text.find(PARAMETER_END, start)
            if end == -1:
                raise ValueError(f"missing closing tag for parameter {key!r}")
            raw = text[start:end]
            # Remove one line break added by the template, or the inline separator.
            if raw.startswith(("\n", "\r\n")):
                raw = re.sub(r"^\r?\n|\r?\n\Z", "", raw)
            else:
                raw = raw.strip()
            value = decode_argument(raw, schema)

        arguments[key] = value
        position = skip_whitespace(text, end + len(PARAMETER_END))
    return arguments, position


def parse_call_group(text, start, schemas):
    position = skip_whitespace(text, start.end())
    calls = []
    while function := FUNCTION_START.match(text, position):
        name = function[1]
        arguments, end = parse_parameters(text, function.end(), schemas.get(name, {}))
        if not text.startswith("</function>", end):
            raise ValueError(f"malformed tool call to {name!r}")
        calls.append({"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)})
        position = skip_whitespace(text, end + len("</function>"))
    if not calls:
        raise ValueError("missing function name in tool call")
    if text.startswith("</tool_call>", position):
        return calls, position + len("</tool_call>")
    # A complete function block can also appear without the outer wrapper.
    return calls, end + len("</function>")


def split_calls(content, schemas, warnings):
    calls, text = [], []
    cursor = search_from = 0
    while start := CALL_START.search(content, search_from):
        try:
            group, end = parse_call_group(content, start, schemas)
        except ValueError as exc:
            warnings.append(str(exc))
            # Preserve the malformed block as text and skip over its internal tags.
            close = content.find("</tool_call>", start.end())
            search_from = close + len("</tool_call>") if close != -1 else len(content)
            continue
        text.append(content[cursor:start.start()])
        calls.extend(group)
        cursor = search_from = end

    text.append(content[cursor:])
    return "".join(text).strip(), calls


def expand_messages(messages):
    for message in messages:
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        parts = CHAT_BOUNDARY.split(content) if role == "user" else [content]
        first = parts[0]
        # Some exports omit the first assistant boundary in a flattened transcript.
        if (len(parts) > 1 and first.rstrip().endswith("</tool_call>")
                and parts[2].lstrip().startswith("<tool_response>")):
            user, think, assistant = first.partition("<think>")
            if think and "</think>" in assistant:
                yield "user", user
                first, role = think + assistant, "assistant"
        yield role, first
        yield from zip(parts[1::2], parts[2::2])


def is_tool_response(message):
    role, content = message
    return role == "user" and content.lstrip().startswith("<tool_response>")


def group_tool_responses(messages):
    for tool_responses, group in groupby(messages, key=is_tool_response):
        if tool_responses:
            yield "user", "\n".join(content for _, content in group)
        else:
            yield from group


def split_responses(content):
    position = skip_whitespace(content, 0)
    while response := RESPONSE.match(content, position):
        yield response[1]
        position = skip_whitespace(content, response.end())
        divider = RESPONSE_SEPARATOR.match(content, position)
        if divider:
            position = divider.end()
    if position < len(content):
        raise ValueError("malformed tool response")


def match_response(pending_calls, reminder):
    if reminder:
        response_id, name = reminder.groups()
        for call in pending_calls:
            if call["function"]["name"] == name:
                pending_calls.remove(call)
                call["id"] = response_id
                return call
    return pending_calls.popleft()


def convert_record(record, index, warnings=None):
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if record.get("images"):
        raise ValueError("multimodal records are unsupported")

    warnings = warnings if warnings is not None else []
    conversations, tools = [], []
    schemas = {}
    pending_calls = deque()
    call_index = 0
    messages = list(group_tool_responses(expand_messages(messages)))

    for offset, (role, content) in enumerate(messages):
        if role == "system" and offset == 0:
            content, tools = split_system(content)
            schemas = {
                tool["function"]["name"]: tool["function"].get("parameters", {}).get(
                    "properties", {}
                )
                for tool in tools
            }
        elif is_tool_response((role, content)):
            responses = list(split_responses(content))
            reminders = []
            if offset + 1 < len(messages):
                next_role, next_content = messages[offset + 1]
                if next_role == "user" and RESPONSE_ID.match(next_content.lstrip()):
                    reminders = list(RESPONSE_ID.finditer(next_content))
            if len(reminders) != len(responses):
                reminders = [None] * len(responses)
            for response, reminder in zip(responses, reminders):
                if pending_calls:
                    call = match_response(pending_calls, reminder)
                    conversations.append({
                        "role": "tool",
                        "content": response,
                        "tool_call_id": call["id"],
                    })
                else:
                    warnings.append(f"message {offset + 1}: unmatched tool response retained as user text")
                    conversations.append({
                        "role": "user",
                        "content": f"<tool_response>\n{response}\n</tool_response>",
                    })
            continue
        elif role not in ("assistant", "user"):
            raise ValueError(f"message {offset + 1}: unexpected role {role!r}")

        if role == "assistant" or not content.lstrip().startswith("<system-reminder>"):
            pending_calls.clear()
        turn = {"role": role, "content": content}
        if role == "assistant":
            content = content.lstrip()
            if content.startswith("<think>"):
                reasoning, _, content = content[len("<think>"):].partition("</think>")
                turn["reasoning_content"] = reasoning.strip()

            call_warnings = []
            turn["content"], calls = split_calls(content, schemas, call_warnings)
            warnings.extend(f"message {offset + 1}: {warning}" for warning in call_warnings)
            if calls:
                turn["tool_calls"] = []
                for function in calls:
                    call_index += 1
                    call = {
                        "id": f"call_{index:05d}_{call_index}",
                        "type": "function",
                        "function": function,
                    }
                    pending_calls.append(call)
                    turn["tool_calls"].append(call)
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
        warnings = []
        record = convert_record(json.loads(line), index, warnings)
        return json.dumps(record, ensure_ascii=False), [
            f"input line {line_number} (record {index}): {warning}" for warning in warnings
        ]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"input line {line_number} (record {index}): {exc}") from exc


def write_batch(results, output):
    for line, warnings in results:
        output.write(line + "\n")
        for warning in warnings:
            print(warning, file=sys.stderr)


def write_records(source, output, workers):
    records = read_records(source)
    if workers == 1:
        write_batch(map(convert_line, records), output)
        return

    # Bound queued work while keeping input order and stable IDs.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while batch := list(islice(records, 32 * workers)):
            write_batch(pool.map(convert_line, batch, chunksize=4), output)


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
