# Reproducibility guide

The release has two reproduction levels: a CPU-only artifact check for the exact paper-facing selector result, and full GPU inference for regenerating candidate pools.

## 1. Exact selector result on CPU

Use Python 3.10 or 3.11 in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt

python scripts/evaluate_selection_on_candidates.py \
  --candidates_path artifacts/candidate_pools/taskbench_dream_k5_test334.json \
  --model_path artifacts/value_function/plan_scorer_combo07_toolset.pkl \
  --bootstrap 1000 \
  --seed 0 \
  --out_json results/taskbench_selection_eval.json
```

Expected values rounded to three decimals:

| Selector | ToolF1 | EdgeRec |
|---|---:|---:|
| K=1 | 0.685 | 0.244 |
| Heuristic | 0.690 | 0.311 |
| Value function | 0.716 | 0.309 |
| Oracle | 0.768 | 0.283 |

The model uses deployable tool-set features, a gradient-boosting regressor, and the target `0.7 × ToolF1 + 0.3 × EdgeRec`. Its training candidate pool is disjoint from the 334 test samples.

## 2. Regenerate TaskBench candidates

Install `requirements-inference.txt`, ensure the model licenses and hardware requirements are acceptable, and run:

```bash
python scripts/collect_candidate_data_v2.py \
  --dlm_path Dream-org/Dream-v0-Instruct-7B \
  --ar_path Qwen/Qwen2.5-7B-Instruct \
  --data_path data/taskbench/taskbench_hf_improved_flattened.jsonl \
  --ids_file data/ids_500.txt \
  --output_path results/taskbench_dream_k5.json \
  --device cuda:0 \
  --K 5 \
  --max_single 167 \
  --max_chain 167 \
  --max_dag 167 \
  --dlm_steps 128 \
  --seed 42
```

The proposer and refiner are loaded in separate phases to reduce peak memory. Exact text generations can still vary across CUDA, driver, GPU, and library versions; the released candidate pool is therefore the canonical artifact for checking the reported selector table.

Train a new value function with:

```bash
python scripts/train_plan_scorer_v3.py \
  --data_path artifacts/candidate_pools/taskbench_dream_k5_train670.json \
  --output_path results/plan_scorer_combo07_toolset.pkl \
  --model_type gbm \
  --feature_set toolset \
  --label combo \
  --combo_alpha 0.7 \
  --seed 42
```

Alternative proposer entry points are:

```text
scripts/collect_candidate_data_arproposer_v1.py   Qwen sampling
scripts/collect_candidate_data_arbeam_v1.py       Qwen beam search
scripts/collect_candidate_data_llada_v1.py        LLaDA-8B
scripts/collect_candidate_data_llada2_v1.py       LLaDA2.0
```

Each script exposes its generation controls through `--help`. Sharded runs can be combined with `scripts/merge_candidate_shards.py`.

## 3. Controlled early-commitment study

Generate the fixed 23-bit tool-set problem and train matched AR and masked-denoising Transformer backbones:

```bash
python scripts/playground_toolset_gen.py \
  --output_dir data/playground_toolset_v2 \
  --seed 42

python scripts/playground_toolset_train_ar.py \
  --train_path data/playground_toolset_v2/train.jsonl \
  --valid_path data/playground_toolset_v2/valid.jsonl \
  --output_dir experiments/playground_toolset_v2_ar \
  --device cuda:0 \
  --seed 42

python scripts/playground_toolset_train_md.py \
  --train_path data/playground_toolset_v2/train.jsonl \
  --valid_path data/playground_toolset_v2/valid.jsonl \
  --output_dir experiments/playground_toolset_v2_md \
  --device cuda:0 \
  --seed 42
```

Evaluate both checkpoints with sampling seeds `0 1 2 3 4 5`:

```bash
mkdir -p results/playground_replicates
for seed in 0 1 2 3 4 5; do
  python scripts/playground_toolset_eval.py \
    --test_path data/playground_toolset_v2/test.jsonl \
    --ar_ckpt experiments/playground_toolset_v2_ar/toolset_ar_best.pt \
    --md_ckpt experiments/playground_toolset_v2_md/toolset_md_best.pt \
    --device cuda:0 \
    --Kmax 10 \
    --seed "$seed" \
    --out_json "results/playground_replicates/rep_${seed}.json"
done

python scripts/summarize_playground_toolset_replicates.py \
  --in_dir results/playground_replicates \
  --out_md results/playground_replicates/summary.md \
  --k 10
```

Generated datasets, checkpoints, and results are intentionally ignored; only the code needed to reproduce them is released.

## 4. Integrity checks

```bash
python -m compileall -q scripts tests
python -m pytest -q
python scripts/audit_public_release.py
```

CI runs these CPU-safe checks on every push and pull request.
