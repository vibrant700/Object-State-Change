#!/usr/bin/env bash
# 在 AutoDL 上按体积从小到大串跑千问系列，跑完汇总成一张对比表。
#
#   bash run.sh                      # 全部跑
#   bash run.sh 2                    # 只跑第 2 个（Qwen2.5-VL-3B）
#   LIMIT=10 bash run.sh             # 每个模型只跑 10 题（冒烟）
#
# 权重目录按需改：WEIGHTS=/root/autodl-tmp/weights
set -euo pipefail

# 权重在 eval/ 的上一级（与 data/、eval/ 平级），所以默认 WEIGHTS=".."
WEIGHTS="${WEIGHTS:-..}"
QA="${QA:-../data/qa_500_2500items_mcq.json}"
FRAMES="${FRAMES:-../data/os-data-3frame}"
RESULTS="${RESULTS:-results}"
LIMIT="${LIMIT:-}"
BATCH="${BATCH:-4}"

# 顺序：最轻 → 正式基线
MODELS=(
  "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
)

ONLY="${1:-}"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

for i in "${!MODELS[@]}"; do
  n=$((i + 1))
  if [ -n "$ONLY" ] && [ "$ONLY" != "$n" ]; then continue; fi

  model="${MODELS[$i]}"
  short="${model##*/}"
  local_dir="$WEIGHTS/$short"

  if [ ! -d "$local_dir" ]; then
    echo "[skip] $short -- 权重不存在: $local_dir"
    continue
  fi

  echo ""
  echo "============================================================"
  echo "  [$n/${#MODELS[@]}] $short"
  echo "============================================================"
  python eval_mcq.py \
    --model "$model" \
    --local-dir "$local_dir" \
    --qa "$QA" \
    --frames "$FRAMES" \
    --out "$RESULTS/$short" \
    --batch-size "$BATCH" \
    $LIMIT_ARG
done

# 汇总：把每个模型的 summary.json 拍平成一张 CSV
python - "$RESULTS" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/summary.json")):
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "model": summary_path.parent.name,
        "n": s["n"],
        "accuracy": s["accuracy"],
        "random_baseline": s["random_baseline"],
        "parse_error_rate": s["parse_error_rate"],
    }
    for label, acc in s["accuracy_by_answer_label"].items():
        row[f"label::{label}"] = acc
    for cat, acc in s["accuracy_by_question_category"].items():
        row[f"cat::{cat}"] = acc
    rows.append(row)

if not rows:
    sys.exit(f"no summary.json found under {root}")

# Union across all rows -- a model run with --limit may cover fewer
# categories than the others, and its columns must not be dropped.
cols = ["model", "n", "accuracy", "random_baseline", "parse_error_rate"]
seen = {c for r in rows for c in r}
cols += sorted(c for c in seen if c.startswith("label::"))
cols += sorted(c for c in seen if c.startswith("cat::"))

out = root / "summary_all_models.csv"
with out.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})

print(f"\n[info] wrote {out}")
print(f"{'model':<26} {'acc':>7} {'rand':>6}  parse_err")
for r in rows:
    print(f"{r['model']:<26} {r['accuracy']:>7} {r['random_baseline']:>6}  {r['parse_error_rate']}")
PY
