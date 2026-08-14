# Data and artifact provenance

This release keeps the paper-facing inputs small and auditable. It does not include pretrained model weights, model caches, the full upstream benchmark repositories, or raw experiment directories.

## TaskBench

`data/taskbench/taskbench_hf_improved_flattened.jsonl` is a deterministic, paper-compatible reconstruction of the `improved.parquet` file published in the official Microsoft TaskBench dataset. The conversion preserves 7,458 rows and maps the flattened fields expected by the DiG-Plan scripts. Filtering every ground-truth tool against the canonical 23-tool catalog retains 6,496 rows: 2,915 single-tool, 2,936 chain, and 645 DAG samples.

The reconstructed JSONL is suitable for this release's scripts, but it is not claimed to be byte-identical to every historical TaskBench checkout. Recreate it from the official source with:

```bash
mkdir -p data/raw/taskbench/data_huggingface
curl -L \
  https://huggingface.co/datasets/microsoft/Taskbench/resolve/main/data_huggingface/improved.parquet \
  -o data/raw/taskbench/data_huggingface/improved.parquet
curl -L \
  https://huggingface.co/datasets/microsoft/Taskbench/resolve/main/data_huggingface/tool_desc.json \
  -o data/raw/taskbench/data_huggingface/tool_desc.json

python scripts/build_taskbench_paper_flattened_from_parquet.py
```

The historical filename `data/ids_500.txt` contains 501 IDs: 167 each for single, chain, and DAG tasks. The main compositional evaluation uses its 334 chain and DAG IDs. `data/splits/ijcai_test_ids.txt` is the same 501-ID balanced set, while `data/splits/ijcai_train_ids_chain450_dag450_seed42.txt` contains a disjoint 900-ID training split.

TaskBench is distributed under the MIT License. The upstream license is copied to `licenses/MICROSOFT_JARVIS_MIT.txt`; see `THIRD_PARTY_NOTICES.md`.

## API-Bank

API-Bank content is not redistributed here. Clone or download the official DAMO-ConvAI release into the ignored `data/raw/` area and convert it locally:

```bash
git clone --depth 1 https://github.com/AlibabaResearch/DAMO-ConvAI.git data/raw/DAMO-ConvAI

python scripts/build_apibank_taskbench_format.py \
  --samples_root data/raw/DAMO-ConvAI/api-bank/lv1-lv2-samples/level-1-given-desc \
  --apis_dir data/raw/DAMO-ConvAI/api-bank/apis \
  --out_dir data/processed/api-bank/level-1-given-desc
```

The released API-Bank files under `data/splits/` contain sample IDs only. The upstream DAMO-ConvAI MIT license is copied to `licenses/ALIBABA_DAMO_CONVAI_MIT.txt`.

## Candidate pools and value function

- `artifacts/candidate_pools/taskbench_dream_k5_train670.json` contains 3,350 candidates from 670 training samples. Its sample IDs are disjoint from the released test split.
- `artifacts/candidate_pools/taskbench_dream_k5_test334.json` contains 1,670 candidates from the 334 held-out compositional samples.
- `artifacts/value_function/plan_scorer_combo07_toolset.pkl` is the exact scikit-learn value-function bundle used by the CPU reproduction command.
- `artifacts/results/taskbench_selection_eval.json` records the corresponding per-sample and aggregate selection result.

Candidate artifacts contain sample IDs, proposed tools/edges, confidence and heuristic features, and evaluation metrics. They do not contain natural-language requests or user data. The pickle must only be loaded from a trusted copy; `requirements-analysis.txt` pins the exact validated scikit-learn runtime.

## SHA-256 checksums

```text
113fda5637517d4bfc5bb2d64c6fa754e479116f3aa4ba0d6e419f07032cd23a  data/taskbench/taskbench_hf_improved_flattened.jsonl
084f973b75cbca5bfb139ebbd820827876b0066913fd259147bd16510ec643f3  data/ids_500.txt
f688be0ebe3a54893ba93b8e324ab915c672eb3ef2d854a8f27caa7de6473189  artifacts/candidate_pools/taskbench_dream_k5_train670.json
0e40ec04df89e411043f4e605233a99e788684d107a7da9c0c18b8b8b66779fa  artifacts/candidate_pools/taskbench_dream_k5_test334.json
5d058a0ca3f7f0f3eb60773007da33ccbb4efd13618d04a0de654a12ae40cdc4  artifacts/value_function/plan_scorer_combo07_toolset.pkl
bb97d6817053fb8f95df554b4ca03b2f37af8b035b149d51dd159043bf6311a2  artifacts/results/taskbench_selection_eval.json
```

Verify them from the repository root with `sha256sum`.
