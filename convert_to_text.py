#!/usr/bin/env python3
"""Pack rendered messages into {id, text} JSONL without a generation prefix.

Usage: python3 convert_to_text.py origin.jsonl text.jsonl --workers 4
The input content already contains thinking and tool markup. Preserve that markup
and strip only surrounding message whitespace, as in the issue's target example.
IDs use the source id, then uuid, then distill_data_00001-style record numbering.
"""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from convert_distill_data import write_records


def convert_record(record, index):
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if record.get("images") or record.get("tools"):
        raise ValueError("images and tools must already be represented in message content")

    text = []
    for number, message in enumerate(messages, 1):
        role, content = message["role"], message["content"]
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"message {number}: unexpected role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"message {number}: content must be text")
        if message.get("tool_calls") or message.get("reasoning_content"):
            raise ValueError(f"message {number}: thinking and tool calls must already be in content")
        text.append(f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n")

    record_id = record.get("id") or record.get("uuid") or f"distill_data_{index:05d}"
    if not isinstance(record_id, str):
        raise ValueError("id or uuid must be a string")
    return {"id": record_id, "text": "".join(text)}


def convert_line(task):
    index, line_number, line = task
    try:
        result = convert_record(json.loads(line), index)
        return json.dumps(result, ensure_ascii=False), []
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"input line {line_number} (record {index}): {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source JSONL or indented JSON object")
    parser.add_argument("output", type=Path, help="destination {id, text} JSONL")
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
            write_records(source, output, args.workers, converter=convert_line)
        temporary.replace(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
