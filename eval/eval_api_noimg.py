#!/usr/bin/env python3
"""Text-only (no-image) control for eval_api.py.

What it answers: how much of the MCQ accuracy comes from the question + options
text ALONE, with no images at all.

  * If accuracy drops to ~20% (chance) -> the images were doing the work, and
    the visual numbers (7B 52.6% / Qwen-VL-Max 59.8%) are real visual reasoning.
  * If it stays high -> a large chunk of the headline number is text prior, not
    vision, and the result must be re-read accordingly.

Same prompt parsing / resume / summary format as eval_api.py, so the numbers
compare directly against results/qwen-vl-max.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from eval_mcq import load_items, parse_answer, print_summary, summarize

# Neutral prompt -- deliberately NO mention of images, so the model is not
# "missing" visual context it has been told about; it just answers from text.
TEXT_ONLY_TEMPLATE = """Question: {question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

Reply with only the letter of the correct option (A, B, C, D, or E)."""

BACKENDS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env": "DASHSCOPE_API_KEY",
    },
    "openai": {"base_url": None, "env": "OPENAI_API_KEY"},
}


def build_messages(item):
    # text only -- deliberately no images
    return [{"role": "user", "content": item["prompt"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--qa", default="../data/qa_500_2500items_mcq.json")
    ap.add_argument("--frames", default="../data/os-data-3frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")

    cfg = BACKENDS[args.backend]
    api_key = os.environ.get(cfg["env"])
    if not api_key:
        sys.exit(f"set {cfg['env']} first")
    client = OpenAI(api_key=api_key, **({"base_url": cfg["base_url"]} if cfg["base_url"] else {}))

    items = load_items(Path(args.qa), Path(args.frames), args.limit)
    if not items:
        sys.exit("no questions loaded -- check --qa and --frames")
    for it in items:
        it["prompt"] = TEXT_ONLY_TEMPLATE.format(question=it["question"], **it["options"])
    print(f"[info] {len(items)} questions loaded (text-only, no images)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        done.add(json.loads(line)["qa_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    pending = [it for it in items if it["qa_id"] not in done]
    if done:
        print(f"[info] resuming -- {len(done)} already scored, {len(pending)} to go")

    failures = 0
    t0 = time.time()
    with pred_path.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(pending, 1):
            raw, last_err = "", None
            for attempt in range(args.retries):
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=build_messages(item),
                        max_tokens=args.max_new_tokens,
                        temperature=0,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(2 ** attempt)
            if last_err is not None:
                failures += 1
                print(f"[warn] {item['qa_id']} failed: {last_err}", file=sys.stderr)

            predicted, is_parse_error = parse_answer(raw)
            fh.write(json.dumps({
                "qa_id": item["qa_id"],
                "clip_id": item["clip_id"],
                "question_category": item["question_category"],
                "answer_label": item["answer_label"],
                "question": item["question"],
                "correct_option": item["correct_option"],
                "predicted": predicted,
                "is_correct": predicted == item["correct_option"],
                "is_parse_error": is_parse_error,
                "raw_output": raw,
            }, ensure_ascii=False) + "\n")
            fh.flush()

            if i % 25 == 0:
                rate = (time.time() - t0) / i
                print(f"[info] {i}/{len(pending)} ({rate:.2f}s/item)")
            if args.sleep:
                time.sleep(args.sleep)

    rows = []
    with pred_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    summary = summarize(rows)
    summary["model"] = args.model
    summary["backend"] = args.backend
    summary["mode"] = "text_only_no_images"
    summary["request_failures"] = failures
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary)
    print(f"[info] wrote {pred_path} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
