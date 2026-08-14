#!/usr/bin/env python3
"""
Retriever Baseline 实验
对比: DLM Proposer vs Dense Retriever

核心论点: Diffusion Proposer > Dense Retriever
- Retriever 只看语义相似度，不懂任务逻辑
- Diffusion 能理解任务间的结构关系

Baseline:
1. Retriever + AR: 用 Sentence-BERT 检索 Top-K 工具 → AR 连边
2. DLM + AR (Hybrid): DLM 选工具 → AR 连边
"""

import json
import argparse
import time
from typing import List, Dict, Tuple
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer

from tool_catalog import (
    get_canonical_tool_definitions,
    filter_samples_by_tools,
)


def build_edge_construction_prompt(instruction: str, selected_tools: List[str], tool_list: List[Dict]) -> str:
    """构建边构建 prompt (给 AR)"""
    tool_desc = {}
    for tool in tool_list:
        if tool['name'] in selected_tools:
            tool_desc[tool['name']] = tool.get('description', 'No description')

    tool_string = "# SELECTED TOOLS #:\n"
    for name in selected_tools:
        desc = tool_desc.get(name, 'No description')
        tool_string += f"- {name}: {desc}\n"

    prompt = f"""{tool_string}
# USER REQUEST #: {instruction}

# TASK #: Based on the selected tools above, generate a complete task plan.
The format must be a strict JSON:
{{
  "task_steps": ["step descriptions"],
  "task_nodes": [{{"task": "tool name", "arguments": []}}],
  "task_links": [{{"source": "tool_i", "target": "tool_j"}}]
}}

IMPORTANT:
- Use ONLY the tools listed above
- task_links should reflect the correct execution order and dependencies

# RESULT #:"""
    return prompt


def build_tool_selection_prompt(instruction: str, tool_list: List[Dict]) -> str:
    """构建工具选择 prompt (给 DLM)"""
    tool_string = "# AVAILABLE TOOLS #:\n"
    for tool in tool_list:
        tool_string += f"- {tool['name']}: {tool.get('description', 'No description')}\n"

    prompt = f"""{tool_string}
# USER REQUEST #: {instruction}

# TASK #: Select the tools needed to complete this request.
Output ONLY the tool names as a JSON list, e.g., ["tool1", "tool2", "tool3"]

# SELECTED TOOLS #:"""
    return prompt


def extract_tools_from_text(text: str, valid_tools: set) -> List[str]:
    """从生成文本中提取工具"""
    tools = []
    import re
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # 尝试解析为 JSON list
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start >= 0 and end > start:
            tool_list = json.loads(text[start:end+1])
            if isinstance(tool_list, list):
                for t in tool_list:
                    if isinstance(t, str) and t in valid_tools:
                        tools.append(t)
                return tools
    except:
        pass

    # 尝试解析完整 JSON
    start = text.find('{')
    if start >= 0:
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        try:
            result = json.loads(text[start:end+1])
            nodes = result.get('task_nodes', [])
            for node in nodes:
                if isinstance(node, dict):
                    task = node.get('task', '')
                    if task in valid_tools:
                        tools.append(task)
            return tools
        except:
            pass

    # 关键词匹配
    for tool in valid_tools:
        if tool.lower() in text.lower():
            tools.append(tool)

    return list(set(tools))


def extract_edges_from_text(text: str) -> List[Tuple[str, str]]:
    """从生成文本中提取边"""
    edges = []
    import re
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    start = text.find('{')
    if start >= 0:
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        try:
            result = json.loads(text[start:end+1])
            links = result.get('task_links', [])
            for link in links:
                if isinstance(link, dict):
                    src = link.get('source', '')
                    tgt = link.get('target', '')
                    if src and tgt:
                        edges.append((src, tgt))
        except:
            pass

    return edges


def compute_metrics(pred_tools: List[str], gt_tools: List[str],
                   pred_edges: List[Tuple[str, str]], gt_edges: List[Tuple[str, str]]) -> Dict:
    """计算评估指标"""
    pred_tool_set = set(pred_tools)
    gt_tool_set = set(gt_tools)

    # Tool F1
    if not pred_tool_set and not gt_tool_set:
        tool_f1 = 1.0
    elif not pred_tool_set or not gt_tool_set:
        tool_f1 = 0.0
    else:
        tp = len(gt_tool_set & pred_tool_set)
        precision = tp / len(pred_tool_set)
        recall = tp / len(gt_tool_set)
        tool_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Tool Recall (重要：衡量工具覆盖)
    tool_recall = len(gt_tool_set & pred_tool_set) / len(gt_tool_set) if gt_tool_set else 1.0

    # Edge Recall
    pred_edge_set = set(pred_edges)
    gt_edge_set = set(gt_edges)
    if not gt_edge_set:
        edge_recall = 1.0 if not pred_edge_set else 0.0
    else:
        edge_recall = len(gt_edge_set & pred_edge_set) / len(gt_edge_set)

    return {
        "tool_f1": tool_f1,
        "tool_recall": tool_recall,
        "edge_recall": edge_recall,
        "n_pred_tools": len(pred_tool_set),
        "n_gt_tools": len(gt_tool_set),
    }


class RetrieverProposer:
    """使用 Dense Retriever 选择工具"""

    def __init__(self, tool_list: List[Dict], device: str = "cuda"):
        print("Loading Sentence-BERT for retrieval...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.tool_list = tool_list
        self.tool_names = [t['name'] for t in tool_list]

        # 预计算工具 embeddings
        tool_texts = [f"{t['name']}: {t.get('description', '')}" for t in tool_list]
        self.tool_embeddings = self.model.encode(tool_texts, convert_to_tensor=True)
        print(f"Indexed {len(tool_list)} tools")

    def retrieve_tools(self, query: str, top_k: int = 5) -> List[str]:
        """检索 Top-K 相关工具"""
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # 计算余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.tool_embeddings
        )

        # 获取 Top-K
        top_indices = similarities.argsort(descending=True)[:top_k]
        return [self.tool_names[i] for i in top_indices.cpu().numpy()]


class DLMProposer:
    """DLM 工具选择器"""

    def __init__(self, model_path: str, device: str):
        self.device = device
        print(f"Loading DLM from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("DLM loaded.")

    def select_tools(self, prompt: str, max_tokens: int = 128, steps: int = 128) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        output = self.model.diffusion_generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            steps=steps,
            top_p=0.95,
            alg="entropy"
        )
        return self.tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)


class ARRefiner:
    """AR 边构建器"""

    def __init__(self, model_path: str, device: str):
        self.device = device
        print(f"Loading AR from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("AR loaded.")

    def build_plan(self, prompt: str, max_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)


def run_retriever_baseline(
    dlm_path: str,
    ar_path: str,
    data_path: str,
    output_path: str,
    device: str,
    ids_file: str = None,
    max_samples: int = 100,
    top_k: int = 5
):
    print(f"\n{'='*60}")
    print(f"Retriever vs DLM Baseline Comparison")
    print(f"{'='*60}\n")

    # 加载数据
    with open(data_path, 'r') as f:
        all_samples = [json.loads(line) for line in f]

    tool_list = get_canonical_tool_definitions()
    tool_names = [t["name"] for t in tool_list]
    valid_tools = set(tool_names)
    filtered_samples = filter_samples_by_tools(all_samples, valid_tools)
    if not filtered_samples:
        raise ValueError("No samples remain after filtering with canonical tool set.")
    print(f"Total canonical tools: {len(valid_tools)}")
    print(f"Filtered samples: {len(filtered_samples)} / {len(all_samples)}")

    # 采样逻辑：如果提供了 ids_file，则只跑这些样本；否则使用原有分层采样
    import random
    random.seed(42)

    if ids_file:
        with open(ids_file, "r") as f:
            target_ids = set(line.strip() for line in f if line.strip())
        samples = [s for s in filtered_samples if s.get("id", "") in target_ids]
        print(f"Loaded {len(target_ids)} IDs from {ids_file}, matched {len(samples)} samples")
    else:
        dag_samples = [s for s in filtered_samples if s.get('type') == 'dag']
        chain_samples = [s for s in filtered_samples if s.get('type') == 'chain']
        single_samples = [s for s in filtered_samples if s.get('type') == 'single']

        # 按比例采样
        n_dag = min(len(dag_samples), max(4, max_samples // 10))
        n_chain = min(len(chain_samples), max_samples // 2)
        n_single = min(len(single_samples), max_samples - n_dag - n_chain)

        random.shuffle(dag_samples)
        random.shuffle(chain_samples)
        random.shuffle(single_samples)

        samples = dag_samples[:n_dag] + chain_samples[:n_chain] + single_samples[:n_single]
        samples = samples[:max_samples]
        print(f"Evaluating {len(samples)} samples (DAG:{n_dag}, Chain:{n_chain}, Single:{n_single})")

    # 初始化模型
    retriever = RetrieverProposer(tool_list, device)
    dlm = DLMProposer(dlm_path, device)
    ar = ARRefiner(ar_path, device)

    results = {
        "retriever_ar": [],   # Retriever + AR
        "dlm_ar": [],         # DLM + AR (Hybrid)
    }

    for i, sample in enumerate(samples):
        instruction = sample.get('instruction', '')
        gt_nodes = json.loads(sample.get('tool_nodes', '[]'))
        gt_tools = [n.get('task', '') for n in gt_nodes]
        gt_links = json.loads(sample.get('tool_links', '[]'))
        gt_edges = [(l.get('source', ''), l.get('target', '')) for l in gt_links]
        task_type = sample.get('type', 'unknown')

        # === 方法 1: Retriever + AR ===
        retrieved_tools = retriever.retrieve_tools(instruction, top_k=top_k)
        edge_prompt = build_edge_construction_prompt(instruction, retrieved_tools, tool_list)
        retriever_output = ar.build_plan(edge_prompt)
        retriever_pred_tools = extract_tools_from_text(retriever_output, valid_tools)
        retriever_edges = extract_edges_from_text(retriever_output)
        retriever_metrics = compute_metrics(retriever_pred_tools, gt_tools, retriever_edges, gt_edges)

        # === 方法 2: DLM + AR (Hybrid) ===
        tool_prompt = build_tool_selection_prompt(instruction, tool_list)
        dlm_output = dlm.select_tools(tool_prompt)
        dlm_selected = extract_tools_from_text(dlm_output, valid_tools)

        if dlm_selected:
            edge_prompt = build_edge_construction_prompt(instruction, dlm_selected, tool_list)
            hybrid_output = ar.build_plan(edge_prompt)
            hybrid_tools = extract_tools_from_text(hybrid_output, valid_tools)
            hybrid_edges = extract_edges_from_text(hybrid_output)
        else:
            hybrid_tools = []
            hybrid_edges = []

        hybrid_metrics = compute_metrics(hybrid_tools, gt_tools, hybrid_edges, gt_edges)

        print(f"[{i+1}/{len(samples)}] {task_type}: "
              f"Retriever F1={retriever_metrics['tool_f1']:.2f}/Recall={retriever_metrics['tool_recall']:.2f} | "
              f"DLM F1={hybrid_metrics['tool_f1']:.2f}/Recall={hybrid_metrics['tool_recall']:.2f}")

        results["retriever_ar"].append({
            "sample_id": sample.get('id', str(i)),
            "task_type": task_type,
            "retrieved_tools": retrieved_tools,
            **retriever_metrics
        })
        results["dlm_ar"].append({
            "sample_id": sample.get('id', str(i)),
            "task_type": task_type,
            "dlm_selected_tools": dlm_selected,
            **hybrid_metrics
        })

    # 汇总
    summary = {}
    for method in ["retriever_ar", "dlm_ar"]:
        summary[method] = {
            "avg_tool_f1": np.mean([r["tool_f1"] for r in results[method]]),
            "avg_tool_recall": np.mean([r["tool_recall"] for r in results[method]]),
            "avg_edge_recall": np.mean([r["edge_recall"] for r in results[method]]),
        }

        # 按任务类型
        by_type = {}
        for task_type in ["single", "chain", "dag"]:
            type_results = [r for r in results[method] if r["task_type"] == task_type]
            if type_results:
                by_type[task_type] = {
                    "n": len(type_results),
                    "tool_f1": np.mean([r["tool_f1"] for r in type_results]),
                    "tool_recall": np.mean([r["tool_recall"] for r in type_results]),
                }
        summary[method]["by_type"] = by_type

    # 保存
    output_data = {"summary": summary, "results": results, "config": {"top_k": top_k, "n_samples": len(samples)}}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"RETRIEVER vs DLM BASELINE COMPARISON")
    print(f"{'='*60}")

    print(f"\n{'Method':<15} {'Tool F1':<12} {'Tool Recall':<12} {'Edge Recall':<12}")
    print("-" * 50)
    for method in ["retriever_ar", "dlm_ar"]:
        print(f"{method:<15} {summary[method]['avg_tool_f1']:<12.3f} "
              f"{summary[method]['avg_tool_recall']:<12.3f} {summary[method]['avg_edge_recall']:<12.3f}")

    # 计算提升
    improvement = (summary['dlm_ar']['avg_tool_f1'] - summary['retriever_ar']['avg_tool_f1']) / summary['retriever_ar']['avg_tool_f1'] * 100 if summary['retriever_ar']['avg_tool_f1'] > 0 else 0
    print(f"\nDLM vs Retriever: {improvement:+.1f}% Tool F1")

    print(f"\nResults saved to: {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlm_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--ar_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    parser.add_argument("--output_path", default="results/baselines/retriever_vs_dlm.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids_file", default=None, help="Optional file with sample IDs to evaluate (one per line)")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    run_retriever_baseline(**vars(args))
