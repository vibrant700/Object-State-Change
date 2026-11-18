#!/usr/bin/env python3
"""InternVL3-8B evaluation on OSCaR fine-grained object-state QA.

Uses 4-bit quantization (bitsandbytes) to fit in 12GB VRAM (RTX 3080 Ti).
Reuses eval_mcq.py for data loading, answer parsing, and summary format.

    python eval_internvl.py \
        --model OpenGVLab/InternVL3-8B \
        --local-dir /root/autodl-tmp/weights/InternVL3-8B \
        --out results/InternVL3-8B --max-tiles 12

Re-running the same command resumes (skips already-scored qa_ids).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval_mcq import (
    PROMPT_TEMPLATE, load_items, parse_answer, print_summary, summarize,
)

# --------------------------------------------------------------------------- #
# Official InternVL image preprocessing (from InternVL/internvl_chat)
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Official InternVL dynamic_preprocess: split image into 448x448 tiles."""
    from PIL import Image  # noqa: F811

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

class InternVLRunner:
    def __init__(self, path, max_tiles=12, use_4bit=True):
        import torch
        from PIL import Image  # noqa: F811
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.Image = Image
        self.max_tiles = max_tiles
        self.input_size = 448
        self.transform = build_transform(self.input_size)

        print(f"[info] loading InternVL from {path}  (4-bit={use_4bit})")
        load_kwargs = dict(
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        # Monkey-patch: transformers 5.x looks for all_tied_weights_keys
        # but remote-code models only have _tied_weights_keys.
        # Must apply for BOTH 4-bit and non-4-bit paths.
        import torch.nn as nn

        if not hasattr(nn.Module, "all_tied_weights_keys"):
            nn.Module.all_tied_weights_keys = []

        if use_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(path, **load_kwargs).eval()

    def _encode_frames(self, frames):
        """Encode all frames into a single pixel_values tensor + per-image tile counts."""
        torch = self.torch
        pixel_values_list = []
        num_patches_list = []
        for frame_path in frames:
            img = self.Image.open(frame_path).convert("RGB")
            tiles = dynamic_preprocess(
                img,
                min_num=1,
                max_num=self.max_tiles,
                image_size=self.input_size,
                use_thumbnail=True,
            )
            pv = torch.stack([self.transform(t) for t in tiles])
            pixel_values_list.append(pv)
            num_patches_list.append(pv.size(0))
        all_pixel_values = torch.cat(pixel_values_list, dim=0)
        return all_pixel_values, num_patches_list

    def generate(self, items, batch_size, max_new_tokens):
        """InternVL chat() is one conversation at a time (no batching)."""
        torch = self.torch
        outputs = []
        for item in items:
            pixel_values, num_patches_list = self._encode_frames(item["frames"])
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            question = "<image>\n" * len(item["frames"]) + item["prompt"]

            with torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    {"max_new_tokens": max_new_tokens, "do_sample": False},
                    num_patches_list=num_patches_list,
                )
            outputs.append(response)
            del pixel_values
            torch.cuda.empty_cache()
        return outputs


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="InternVL3 evaluation on OSCaR fine-grained state QA"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--qa", default="../data/qa_500_2500items_mcq.json")
    ap.add_argument("--frames", default="../data/os-data-3frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tiles", type=int, default=12,
                    help="max 448x448 tiles per frame (InternVL dynamic tiling)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="disable 4-bit quantization (needs >= 20GB VRAM)")
    ap.add_argument("--chunk", type=int, default=32,
                    help="print progress + flush every N questions")
    args = ap.parse_args()

    items = load_items(Path(args.qa), Path(args.frames), args.limit)
    if not items:
        sys.exit("no questions loaded")
    for it in items:
        it["prompt"] = PROMPT_TEMPLATE.format(
            num_frames=len(it["frames"]), question=it["question"], **it["options"]
        )
    print(f"[info] {len(items)} questions loaded")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    done = set()
    if pred_path.exists():
        for line in pred_path.open(encoding="utf-8"):
            if line.strip():
                try:
                    done.add(json.loads(line)["qa_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    pending = [it for it in items if it["qa_id"] not in done]
    if done:
        print(f"[info] resuming -- {len(done)} already scored")

    if pending:
        runner = InternVLRunner(
            args.local_dir or args.model,
            max_tiles=args.max_tiles,
            use_4bit=not args.no_4bit,
        )
        t0 = time.time()
        written = 0
        with pred_path.open("a", encoding="utf-8") as fh:
            for start in range(0, len(pending), args.chunk):
                chunk = pending[start : start + args.chunk]
                preds = runner.generate(chunk, 1, args.max_new_tokens)
                for item, raw in zip(chunk, preds):
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
                written += len(chunk)
                per = (time.time() - t0) / written
                left = len(pending) - written
                print(
                    f"[info] {written}/{len(pending)}  {per:.2f}s/q  "
                    f"ETA {per * left / 60:.0f}min",
                    flush=True,
                )

    rows = [json.loads(l) for l in pred_path.open(encoding="utf-8") if l.strip()]
    summary = summarize(rows)
    summary["model"] = args.model
    summary["local_dir"] = args.local_dir
    summary["quantization"] = "4bit" if not args.no_4bit else "none"
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(summary)
    print(f"[info] wrote {pred_path} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
