# Third-party notices

DiG-Plan relies on public datasets and models but does not redistribute pretrained model weights.

## TaskBench / Microsoft JARVIS

The file `data/taskbench/taskbench_hf_improved_flattened.jsonl`, the split IDs under `data/`, and the released TaskBench candidate pools are derived from the TaskBench release in Microsoft JARVIS. TaskBench is distributed under the MIT License. The upstream license is preserved in `licenses/MICROSOFT_JARVIS_MIT.txt`.

- Upstream: https://github.com/microsoft/JARVIS/tree/main/taskbench
- Dataset: https://huggingface.co/datasets/microsoft/Taskbench

## API-Bank / Alibaba DAMO-ConvAI

`scripts/build_apibank_taskbench_format.py` consumes the official API-Bank release. API-Bank data is not copied into this repository. The upstream DAMO-ConvAI license is preserved in `licenses/ALIBABA_DAMO_CONVAI_MIT.txt`.

- Upstream: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank
- Dataset: https://huggingface.co/datasets/liminghao1630/API-Bank

## Pretrained models

Dream, Qwen2.5, LLaDA, LLaDA2.0, and all-MiniLM weights are downloaded directly from their publishers and remain subject to their respective model licenses. Their weights, tokenizer caches, and local symlinks are deliberately excluded from this repository.
