#!/usr/bin/env python3
"""Convert messages to {id, text} JSONL using the model's apply_chat_template.

Requires transformers. Example:
    python3 convert_to_text.py origin.jsonl text.jsonl \
        --model /opt/models/Qwen3.8-27B --workers 4

Existing <think> blocks are moved into reasoning_content before rendering so the
Qwen template emits each block once. All historical reasoning is retained.
An output directory creates <input_stem>_text.jsonl inside that directory.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


tokenizer = None


def load_tokenizer(model):
    from transformers import AutoTokenizer

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)


def prepare_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    prepared = []
    for number, message in enumerate(messages, 1):
        message = deepcopy(message)
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError(f"message {number}: content must be text")
        if message["role"] == "assistant" and content.lstrip().startswith("<think>"):
            reasoning, closing, content = content.lstrip()[len("<think>"):].partition("</think>")
            if not closing:
                raise ValueError(f"message {number}: unclosed thinking block")
            existing = message.get("reasoning_content")
            if existing is not None and existing.strip() != reasoning.strip():
                raise ValueError(f"message {number}: conflicting reasoning_content")
            message.update(content=content, reasoning_content=reasoning.strip())
        # Qwen's template iterates structured tool arguments as a mapping.
        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                function = call["function"]
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
        prepared.append(message)
    return prepared


def convert_line(task):
    index, line_number, line = task
    try:
        record = json.loads(line)
        if record.get("images"):
            raise ValueError("image records require multimodal processing")
        text = tokenizer.apply_chat_template(
            prepare_messages(record["messages"]),
            tools=record.get("tools") or None,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            preserve_thinking=True,
        )
        record_id = record.get("id") or record.get("uuid") or f"distill_data_{index:05d}"
        if not isinstance(record_id, str):
            raise ValueError("id or uuid must be a string")
        return json.dumps({"id": record_id, "text": text}, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"input line {line_number} (record {index}): {exc}") from exc


def read_records(source):
    index = 0
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        index += 1
        # Also accept the indented single-object example from the issue.
        if index == 1 and line.strip() == "{":
            line += source.read()
        yield index, line_number, line


def write_records(source, output, model, workers):
    records = read_records(source)
    if workers == 1:
        load_tokenizer(model)
        for result in map(convert_line, records):
            output.write(result + "\n")
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=load_tokenizer, initargs=(model,),
    ) as pool:
        while batch := list(islice(records, 32 * workers)):
            for result in pool.map(convert_line, batch, chunksize=4):
                output.write(result + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="output file or existing directory")
    parser.add_argument("--model", default="/opt/models/Qwen3.8-27B", help="local model directory")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.output.is_dir():
        args.output = args.output / f"{args.input.stem}_text.jsonl"
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    temporary = None
    try:
        with args.input.open(encoding="utf-8") as source, NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            write_records(source, output, args.model, args.workers)
        temporary.replace(args.output)
    except Exception as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
