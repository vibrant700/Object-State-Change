#!/usr/bin/env python3
"""Zero-shot MCQ evaluation of open-source VLMs on the OSCaR fine-grained
object-state QA set.

Each item is egocentric keyframes + one 5-way multiple-choice question.
The model is prompted to emit a single letter; we compare it to `correct_option`
and break accuracy down by `answer_label` and `question_category`.

    python eval_mcq.py --model Qwen/Qwen2.5-VL-3B-Instruct \
                       --local-dir /root/weights/Qwen2.5-VL-3B-Instruct \
                       --qa qa_500_2500items_mcq.json \
                       --frames ../data/os-data-3frame \
                       --out results/Qwen2.5-VL-3B --batch-size 4

Re-running the same command resumes: rows already present in
predictions.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCDE"

PROMPT_TEMPLATE = """The {num_frames} images are frames from a first-person (egocentric) video, shown in chronological order (image 1 = earliest, image {num_frames} = latest).

Question: {question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

Reply with only the letter of the correct option (A, B, C, D, or E)."""

# "Answer: B" / "The answer is B" / bare "B" / "(B)" all collapse to a letter.
_ANSWER_PATTERNS = [
    re.compile(r"answer\s*(?:is|:)?\s*\(?\s*([A-E])\b", re.I),
    re.compile(r"^\s*\(?\s*([A-E])\s*\)?\s*[.),:]?\s*$", re.I),
    re.compile(r"\b([A-E])\b"),
]

NO_ANSWER = ""


def parse_answer(text: str) -> tuple[str, bool]:
    """Return (letter, is_parse_error). Empty letter means we could not read one."""
    if not text:
        return NO_ANSWER, True
    stripped = text.strip()
    for pat in _ANSWER_PATTERNS:
        m = pat.search(stripped)
        if m:
            return m.group(1).upper(), False
    return NO_ANSWER, True


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_items(qa_path: Path, frames_root: Path, limit: int | None):
    """Flatten the nested QA json into one record per question."""
    pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    items = []
    for pair in pairs:
        clip_id = pair["clip_id"]
        clip_dir = frames_root / clip_id
        if not clip_dir.is_dir():
            print(f"[warn] missing frame dir for {clip_id}, skipping", file=sys.stderr)
            continue
        # Sort by the numeric frame index in the filename; indices are not
        # consistently zero-padded across clips.
        frames = sorted(clip_dir.glob("frame_*.jpg"),
                        key=lambda path: int(path.stem.split("_", 1)[1]))
        if not frames:
            print(f"[warn] no frames in {clip_id}, skipping", file=sys.stderr)
            continue
        for qa in pair["qas"]:
            opts = qa["options"]
            if not all(k in opts for k in LETTERS):
                print(f"[warn] {qa['qa_id']} missing options, skipping", file=sys.stderr)
                continue
            items.append(
                {
                    "qa_id": qa["qa_id"],
                    "clip_id": clip_id,
                    "frames": [str(f) for f in frames],
                    "question": qa["question"],
                    "question_category": pair.get("question_category", "unknown"),
                    "answer_label": qa["answer_label"],
                    "options": {k: opts[k] for k in LETTERS},
                    "correct_option": qa["correct_option"].strip().upper(),
                    "grounding_note": qa.get("grounding_note", ""),
                }
            )
    if limit:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------- #
# Qwen2-VL / Qwen2.5-VL / Qwen3-VL
# --------------------------------------------------------------------------- #

def build_qwen_conversation(item):
    """One chat message list with all frames in chronological order."""
    content = [{"type": "image", "image": f"file://{f}"} for f in item["frames"]]
    content.append({"type": "text", "text": item["prompt"]})
    return [{"role": "user", "content": content}]


def _qwen_model_class(name: str):
    """Pick the right Qwen-VL class -- the generations are not interchangeable.

    Qwen2-VL (2B/7B)  -> Qwen2VLForConditionalGeneration
    Qwen2.5-VL (3B/7B)-> Qwen2_5_VLForConditionalGeneration
    Qwen3-VL          -> Qwen3VLForConditionalGeneration
    """
    import transformers

    key = name.lower()
    if "qwen3-vl" in key or "qwen3_vl" in key:
        cls_name = "Qwen3VLForConditionalGeneration"
    elif "qwen2.5-vl" in key or "qwen2_5_vl" in key:
        cls_name = "Qwen2_5_VLForConditionalGeneration"
    else:
        cls_name = "Qwen2VLForConditionalGeneration"

    if not hasattr(transformers, cls_name):
        raise RuntimeError(
            f"{cls_name} not found in transformers {transformers.__version__}; "
            f"upgrade with: pip install -U 'transformers>=4.49.0'"
        )
    return getattr(transformers, cls_name)


class QwenRunner:
    family = "qwen"

    def __init__(self, path, name, dtype, attn, max_pixels, min_pixels):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            path, max_pixels=max_pixels, min_pixels=min_pixels
        )
        # Batched decoder-only generation needs LEFT padding: the new tokens
        # must come after the prompt, and the `out[len(inp):]` trim below
        # assumes pads sit at the front.
        self.processor.tokenizer.padding_side = "left"
        # Detect the generation from the repo id, not the local folder name --
        # users rename local dirs freely.
        model_cls = _qwen_model_class(name)
        print(f"[info] using {model_cls.__name__}")
        self.model = model_cls.from_pretrained(
            path, torch_dtype=dtype, attn_implementation=attn, device_map="cuda:0"
        ).eval()

    def generate(self, items, batch_size, max_new_tokens):
        from qwen_vl_utils import process_vision_info

        outputs = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            convs = [build_qwen_conversation(it) for it in chunk]
            texts = [
                self.processor.apply_chat_template(
                    c, tokenize=False, add_generation_prompt=True
                )
                for c in convs
            ]
            image_inputs, video_inputs = process_vision_info(convs)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda:0")

            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            trimmed = [
                out[len(inp) :]
                for inp, out in zip(inputs.input_ids, generated)
            ]
            decoded = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs.extend(decoded)
            del inputs, generated
            self.torch.cuda.empty_cache()
        return outputs


# --------------------------------------------------------------------------- #
# InternVL2.5 / InternVL3
# --------------------------------------------------------------------------- #

class InternVLRunner:
    family = "internvl"

    def __init__(self, path, name, dtype, attn, max_pixels, min_pixels):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
        self.model = (
            AutoModel.from_pretrained(
                path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=(attn == "flash_attention_2"),
            )
            .eval()
            .cuda()
        )
        # pixel_values are built at native tiling; keep the official default.
        self.max_num = 12
        self.input_size = 448

    def _load_pixels(self, frames):
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        tiles = []
        for frame in frames:
            img = Image.open(frame).convert("RGB")
            # Official dynamic_preprocess: split into 448x448 tiles.
            ratio = 1
            w, h = img.size
            target = self.input_size * ratio
            blocks = max(1, min(self.max_num, round((w * h) / (target * target))))
            cols = max(1, int(round((w / h) ** 0.5 * (blocks**0.5))))
            cols = max(1, min(blocks, cols))
            rows = max(1, blocks // cols)
            resized = img.resize((cols * target, rows * target), Image.BICUBIC)
            for r in range(rows):
                for c in range(cols):
                    box = (
                        c * target,
                        r * target,
                        (c + 1) * target,
                        (r + 1) * target,
                    )
                    tiles.append(transform(resized.crop(box)))
        return torch.stack(tiles)

    def generate(self, items, batch_size, max_new_tokens):
        # InternVL's generate() takes one conversation at a time; batching
        # multi-image inputs through it is not supported by the official API.
        outputs = []
        for item in items:
            pixel_values = self._load_pixels(item["frames"]).to(
                self.torch.bfloat16
            ).cuda()
            question = "<image>\n" * len(item["frames"]) + item["prompt"]
            with self.torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    {"max_new_tokens": max_new_tokens, "do_sample": False},
                )
            outputs.append(response)
            del pixel_values
            self.torch.cuda.empty_cache()
        return outputs


def load_runner(args):
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    path = args.local_dir or args.model
    family = "internvl" if "internvl" in args.model.lower() else "qwen"
    runner_cls = InternVLRunner if family == "internvl" else QwenRunner
    print(f"[info] loading {args.model} ({family}) from {path}")
    t0 = time.time()
    runner = runner_cls(path, args.model, dtype, args.attn, args.max_pixels, args.min_pixels)
    print(f"[info] loaded in {time.time() - t0:.1f}s")
    return runner


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def summarize(rows):
    total = len(rows)
    correct = sum(r["is_correct"] for r in rows)
    parse_errors = sum(r["is_parse_error"] for r in rows)

    def acc(subset):
        if not subset:
            return None
        return round(sum(r["is_correct"] for r in subset) / len(subset), 4)

    by_label = defaultdict(list)
    by_category = defaultdict(list)
    confusion = defaultdict(Counter)
    for r in rows:
        by_label[r["answer_label"]].append(r)
        by_category[r["question_category"]].append(r)
        confusion[r["correct_option"]][r["predicted"]] += 1

    return {
        "n": total,
        "accuracy": acc(rows),
        "random_baseline": round(1 / len(LETTERS), 4),
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / total, 4) if total else None,
        "accuracy_by_answer_label": {k: acc(v) for k, v in sorted(by_label.items())},
        "accuracy_by_question_category": {
            k: acc(v) for k, v in sorted(by_category.items())
        },
        "n_by_answer_label": {k: len(v) for k, v in sorted(by_label.items())},
        "n_by_question_category": {k: len(v) for k, v in sorted(by_category.items())},
        "confusion_matrix": {
            gold: dict(sorted(row.items())) for gold, row in sorted(confusion.items())
        },
    }


def print_summary(summary):
    print(f"\n{'=' * 62}")
    print(f"  questions evaluated : {summary['n']}")
    print(f"  accuracy            : {summary['accuracy']}")
    print(f"  random baseline     : {summary['random_baseline']}")
    print(f"  parse errors        : {summary['parse_errors']} "
          f"({summary['parse_error_rate']})")
    print(f"{'-' * 62}")
    print("  accuracy by answer_label")
    for k, v in summary["accuracy_by_answer_label"].items():
        n = summary["n_by_answer_label"][k]
        print(f"    {k:<14} {v}  (n={n})")
    print(f"{'-' * 62}")
    print("  accuracy by question_category")
    for k, v in summary["accuracy_by_question_category"].items():
        n = summary["n_by_question_category"][k]
        print(f"    {k:<28} {v}  (n={n})")
    print(f"{'=' * 62}\n")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Zero-shot MCQ evaluation on OSCaR fine-grained state QA"
    )
    ap.add_argument("--model", required=True,
                    help="HF/ModelScope repo id or a short label when --local-dir is set")
    ap.add_argument("--local-dir", default=None,
                    help="path to pre-downloaded weights; loads offline")
    ap.add_argument("--qa", default="../data/qa_500_2500items_mcq.json")
    ap.add_argument("--frames", default="../data/os-data-3frame")
    ap.add_argument("--out", required=True, help="output directory for this model run")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only evaluate the first N questions (smoke test)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--max-pixels", type=int, default=2073600,
                    help="Qwen only; caps vision tokens. default = 1920x1080 native")
    ap.add_argument("--min-pixels", type=int, default=3136)
    args = ap.parse_args()

    items = load_items(Path(args.qa), Path(args.frames), args.limit)
    if not items:
        sys.exit("no questions loaded -- check --qa and --frames")
    print(f"[info] {len(items)} questions loaded")

    for it in items:
        it["prompt"] = PROMPT_TEMPLATE.format(
            num_frames=len(it["frames"]), question=it["question"], **it["options"]
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    # Resume: skip questions already scored in a previous run.
    done = set()
    if pred_path.exists():
        with pred_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["qa_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if done:
        print(f"[info] resuming -- {len(done)} questions already scored")
    pending = [it for it in items if it["qa_id"] not in done]

    if pending:
        runner = load_runner(args)
        t0 = time.time()
        preds = runner.generate(pending, args.batch_size, args.max_new_tokens)
        elapsed = time.time() - t0
        print(f"[info] generated {len(preds)} answers in {elapsed:.1f}s "
              f"({elapsed / max(1, len(preds)):.2f}s/question)")

        with pred_path.open("a", encoding="utf-8") as fh:
            for item, raw in zip(pending, preds):
                predicted, is_parse_error = parse_answer(raw)
                fh.write(
                    json.dumps(
                        {
                            "qa_id": item["qa_id"],
                            "clip_id": item["clip_id"],
                            "question_category": item["question_category"],
                            "answer_label": item["answer_label"],
                            "question": item["question"],
                            "correct_option": item["correct_option"],
                            "predicted": predicted,
                            "is_correct": predicted == item["correct_option"],
                            "is_parse_error": is_parse_error,
                            "raw_output": raw.strip(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fh.flush()

    # Re-score the full file so resumed runs report over every question.
    rows = []
    with pred_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    summary = summarize(rows)
    summary["model"] = args.model
    summary["local_dir"] = args.local_dir
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(summary)
    print(f"[info] wrote {pred_path} and {summary_path}")


if __name__ == "__main__":
    main()
