# OSCaR 细粒度 Object State QA —— 可行性验证评测

在 AutoDL 云 GPU 上零样本评测开源 VLM，回答一个问题：

> 现有模型是否能分清物体的**细粒度状态**（切片/切块/完整、半开/全开/关闭……），还是只认得物体？

- 测试集：500 个 clip × 5 道题 = **2500 道五选一 MCQ**
- 每道题输入 3 张第一视角关键帧（时间顺序）
- 指标：**top-1 accuracy**，随机基线 **20%**
- 范式：**zero-shot prompt MCQ**（与 HanDyVQA / EgoPointVQA 一致）

---

## 目录结构

```
eval/
  eval_mcq.py        开源模型评测主脚本（支持断点续跑）
  eval_api.py        API 基线（通义 / OpenAI，可选）
  run.sh             串跑千问系列并汇总成 CSV
  requirements.txt   依赖
  README.md
```

期望的目录布局（权重、`data/`、`eval/` 三者平级）：

```
object-state/
  eval/                                  # 本目录（脚本）
  data/os-data-3frame/<clip_id>/frame_*.jpg
  data/qa_500_2500items_mcq.json
  Qwen2-VL-2B-Instruct/                  # 权重，与 eval/ 平级
```

脚本的 `--qa` / `--frames` 默认值已按此布局设好，**从 `eval/` 目录里运行就不用手敲路径**。

---

## 步骤 1 · 本机先下好权重

按体积从小到大，先下 ① 就能跑通，②③ 可以边跑边下。

| 顺序 | 模型 | 大小 | 魔搭 | HuggingFace |
|---|---|---|---|---|
| ① | Qwen2-VL-2B-Instruct | ~5GB | <https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct> | <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct> |
| ② | Qwen2.5-VL-3B-Instruct | ~7.4GB | <https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct> | <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct> |
| ③ | Qwen2.5-VL-7B-Instruct | ~16GB | <https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct> | <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct> |

用 ModelScope（国内直连）：

```bash
pip install modelscope

modelscope download --model Qwen/Qwen2-VL-2B-Instruct   --local_dir D:\weights\Qwen2-VL-2B-Instruct
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir D:\weights\Qwen2.5-VL-3B-Instruct
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir D:\weights\Qwen2.5-VL-7B-Instruct
```

> ⚠️ `--local_dir` 的目录名要和模型名保持一致，因为 `run.sh` 是按
> `$WEIGHTS/<模型名>` 找权重的。改了名就用 `--local-dir` 手动指。

## 步骤 2 · 开 AutoDL 实例

- GPU：**RTX 4090 24GB**（2B/3B 用 3090 24GB 也够，更便宜）
- 镜像：**PyTorch 2.x + CUDA 12.x**（预置镜像自带 torch，不要重装）
- 数据盘：**≥ 60GB**（权重 28G + 数据 0.7G + 输出）

## 步骤 3 · 上传数据与代码

AutoDL 控制台会给一条 ssh 命令，形如 `ssh -p 12345 root@region-1.autodl.com`。本机执行：

```bash
# 变量按自己的实例改
HOST=region-1.autodl.com
PORT=12345
DEST=/root/autodl-tmp/object-state

ssh -p $PORT root@$HOST "mkdir -p $DEST"

# 代码 + 测试集
scp -P $PORT -r "D:\Object State Change\eval" root@$HOST:$DEST/
scp -P $PORT "D:\Object State Change\output\fine_grained_recaption\qa_500_2500items_mcq.json" root@$HOST:$DEST/

# 关键帧（约 630MB，文件多，建议先打包再传，否则很慢）
#   本机：cd "D:\Object State Change\data" && tar -czf frames.tar.gz os-data-3frame
#   然后：scp -P $PORT frames.tar.gz root@$HOST:$DEST/
#   服务器：cd $DEST && mkdir -p data && tar -xzf frames.tar.gz -C data
```

权重上传（约 28GB，本机上传慢，可以考虑先在云上下 2B 跑通）：

```bash
scp -P $PORT -r "D:\weights\Qwen2-VL-2B-Instruct" root@$HOST:/root/autodl-tmp/weights/
```

## 步骤 4 · 装依赖

```bash
cd /root/autodl-tmp/object-state/eval

# 先确认 torch 能用，打印 True 就跳过 torch 安装
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

pip install -r requirements.txt
```

## 步骤 5 · 冒烟（务必先跑）

```bash
python eval_mcq.py \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --local-dir ../Qwen2-VL-2B-Instruct \
  --out results/smoke \
  --limit 5 --batch-size 1
```

（`--qa` 和 `--frames` 用默认值即可，已指向 `../data/`。）

看三件事：模型加载成功、`raw_output` 是单个字母、`summary.json` 里的 `parse_error_rate` 接近 0。
**如果 `raw_output` 是大段文字**，说明模型没听话，把 `--max-new-tokens` 降到 4 再看。

## 步骤 6 · 全量跑

```bash
bash run.sh
```

`run.sh` 会按 2B → 3B → 7B 顺序跑，每个模型的结果落在 `results/<模型名>/`，
最后汇总出 `results/summary_all_models.csv`。

单个模型跑 / 断点续跑：

```bash
python eval_mcq.py --model Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir ../Qwen2.5-VL-7B-Instruct \
  --out results/Qwen2.5-VL-7B --batch-size 4
```

**同一条命令重跑就是续跑**——已评过的题会自动跳过，云实例中断不用从头来。

---

## 输出怎么看

`results/<模型>/summary.json`：

| 字段 | 含义 |
|---|---|
| `accuracy` | 总体准确率，和 `random_baseline`（0.2）比 |
| `accuracy_by_answer_label` | 五个进度状态（completed / in_progress / not_started / different / uncertain）各自准确率 |
| `accuracy_by_question_category` | 八个状态类别（surface_cleanliness / shape_size_change / …）各自准确率 |
| `confusion_matrix` | 行=正确答案，列=模型预测，看它偏向选哪个 |
| `parse_error_rate` | 没解析出字母的比例，应 < 5% |

**判读可行性验证**：如果 7B 这种强模型总体还在 20%~35% 徘徊（明显低于它在通用 VQA 上的水平），
且 fine-grained 类别（`shape_size_change`、`part_separation`）掉得特别狠，假设就成立。

> ⚠️ **2B/3B 接近 20% 不能单独作为结论**——可能只是模型太小。正式结论必须有 7B 级对照。

---

## 可选 · API 基线

```bash
export DASHSCOPE_API_KEY=sk-xxx        # 通义，国内直连
python eval_api.py --backend dashscope --model qwen-vl-max \
  --qa ../qa_500_2500items_mcq.json --frames ../data/os-data-3frame \
  --out results/qwen-vl-max
```

`--backend openai --model gpt-4o` 同理，需要 `OPENAI_API_KEY` 和梯子。
落盘格式与开源模型完全一致，可直接横向比。

---

## 常见问题

**显存不够 / OOM**：降 `--batch-size`（4 → 2 → 1），或加 `--max-pixels 1003520` 限制视觉 token。

**跑得慢**：先用 `--limit 50` 估速度。3 帧 1920×1080 每张约 2600 个视觉 token，
想把速度换精度就调 `--max-pixels`。

**想用 flash-attention**：`pip install flash-attn --no-build-isolation` 后加 `--attn flash_attention_2`。

**InternVL 系列**：`eval_mcq.py` 里有 `InternVLRunner` 分支，会自动按模型名匹配。
但 InternVL 的官方 chat API 不支持批量多图，**是逐题跑的，会明显慢**，
且需额外装 `timm einops sentencepiece`。建议先用千问系列出结果。
