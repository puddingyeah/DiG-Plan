#!/usr/bin/env python3
"""
AR Refinement 实验
验证 Hybrid 架构: DLM 选工具 → AR 连边

核心假设: DLM 擅长工具选择（多样性），AR 擅长依赖关系构建
"""

import json
import argparse
import time
import signal
import os
from typing import List, Dict, Tuple, Set
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from utils import get_sample_instruction, get_sample_task_links, get_sample_task_nodes


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Generation timed out")


def run_with_timeout(func, timeout_seconds=120):
    """Run a function with timeout (Linux only)"""
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func()
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        print(f"[TIMEOUT] Generation exceeded {timeout_seconds}s, skipping...")
        return ""
    finally:
        signal.signal(signal.SIGALRM, old_handler)

from tool_catalog import (
    get_canonical_tool_definitions,
    filter_samples_by_tools,
)


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


def build_edge_construction_prompt(instruction: str, selected_tools: List[str], tool_list: List[Dict]) -> str:
    """构建边构建 prompt (给 AR)"""
    # 获取选中工具的描述
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
- Each tool should appear exactly once in task_nodes

# RESULT #:"""
    return prompt


def extract_tools_from_text(text: str, valid_tools: set) -> List[str]:
    """从生成文本中提取工具"""
    tools = []
    import re
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # 尝试解析为 JSON list
    try:
        # 找到 [ 和 ] 之间的内容
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
            # Build an index->tool mapping for robustness:
            # Some LMs output links as indices ("0","1") instead of tool names.
            idx_to_tool: List[str] = []
            nodes = result.get("task_nodes", [])
            if isinstance(nodes, list):
                for n in nodes:
                    if isinstance(n, dict):
                        t = n.get("task", "")
                        if isinstance(t, str) and t:
                            idx_to_tool.append(t)
                    elif isinstance(n, str) and n:
                        idx_to_tool.append(n)

            def _maybe_map(x):
                if isinstance(x, int):
                    return idx_to_tool[x] if 0 <= x < len(idx_to_tool) else str(x)
                if isinstance(x, str):
                    xs = x.strip()
                    if xs.isdigit() and idx_to_tool:
                        i = int(xs)
                        if 0 <= i < len(idx_to_tool):
                            return idx_to_tool[i]
                    return xs
                return str(x)

            links = result.get('task_links', [])
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, dict):
                        src = _maybe_map(link.get('source', ''))
                        tgt = _maybe_map(link.get('target', ''))
                        if src and tgt:
                            edges.append((src, tgt))
                    elif isinstance(link, (list, tuple)) and len(link) == 2:
                        src = _maybe_map(link[0])
                        tgt = _maybe_map(link[1])
                        if src and tgt:
                            edges.append((src, tgt))
        except Exception:
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

    # Edge Recall
    pred_edge_set = set(pred_edges)
    gt_edge_set = set(gt_edges)

    if not gt_edge_set:
        edge_recall = 1.0 if not pred_edge_set else 0.0
    else:
        edge_recall = len(gt_edge_set & pred_edge_set) / len(gt_edge_set)

    return {
        "tool_f1": tool_f1,
        "edge_recall": edge_recall,
        "n_pred_tools": len(pred_tool_set),
        "n_gt_tools": len(gt_tool_set),
        "n_pred_edges": len(pred_edge_set),
        "n_gt_edges": len(gt_edge_set)
    }


class DLMToolSelector:
    """DLM 用于工具选择"""

    def __init__(self, model_path: str, device: str):
        self.device = device
        print(f"Loading DLM from {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("DLM loaded.")

    def select_tools(self, prompt: str, max_tokens: int = 256, steps: int = 128, timeout: int = 120) -> str:
        """选择工具 (with timeout protection)"""
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        def generate():
            output = self.model.diffusion_generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_tokens,
                steps=steps,
                top_p=0.95,
                alg="entropy"
            )
            return self.tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

        return run_with_timeout(generate, timeout_seconds=timeout)

    def select_tools_with_confidence(
        self,
        prompt: str,
        max_tokens: int = 256,
        steps: int = 128,
        timeout: int = 120,
    ) -> Tuple[str, Dict[str, List[torch.Tensor]]]:
        """
        选择工具，同时返回一个简单的“不确定性”信号。

        返回:
          - 生成的文本 (与 select_tools 相同风格)
          - 一个字典，其中目前包含:
              {
                "mask_entropies": [Tensor(num_mask_positions_this_step), ...]
              }
            这些是每个去噪 step 上、被 mask 的位置的熵值，可用于后续聚合成 tool-level uncertainty。
        """
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        history: Dict[str, List[torch.Tensor]] = {"mask_entropies": []}

        def logits_hook(step, x, logits):
            """
            Dream 的 generation_logits_hook_func 约定:
              - 必须返回 logits，否则后续流程会拿到 None 而报错。
              - x: 当前 token 序列 (含 mask_token)
              - logits: 对应位置的 logits
            这里我们提取被 mask 的位置对应的 logits，并计算 entropy。
            """
            if logits is None:
                return logits

            # 从 model.config 或 tokenizer 获取 mask_token_id (generation_config 里是 None)
            mask_id = getattr(self.model.config, "mask_token_id", None)
            if mask_id is None:
                mask_id = getattr(self.tokenizer, "mask_token_id", None)
            if mask_id is None:
                return logits

            try:
                mask = (x == mask_id)
            except Exception:
                return logits

            if mask.any():
                mask_logits = logits[mask]  # [num_mask_positions, vocab]
                probs = torch.softmax(mask_logits, dim=-1)
                log_probs = torch.log(probs + 1e-8)
                entropy = -(probs * log_probs).sum(dim=-1)  # [num_mask_positions]
                history["mask_entropies"].append(entropy.detach().cpu())

            return logits

        def generate():
            out = self.model.diffusion_generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_tokens,
                steps=steps,
                top_p=0.95,
                alg="entropy",
                output_history=False,
                return_dict_in_generate=True,
                generation_logits_hook_func=logits_hook,
            )
            # DreamModelOutput.sequences: [B, L]
            sequences = out.sequences
            text = self.tokenizer.decode(
                sequences[0][len(inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )
            return text, history

        return run_with_timeout(generate, timeout_seconds=timeout)


class AREdgeBuilder:
    """AR 用于边构建"""

    def __init__(self, model_path: str, device: str):
        self.device = device
        print(f"Loading AR from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("AR loaded.")

    def build_plan(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        timeout: int = 60,
    ) -> str:
        """构建完整 plan (with timeout protection)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def generate():
            output = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        return run_with_timeout(generate, timeout_seconds=timeout)


def run_ar_refinement_eval(
    dlm_path: str,
    ar_path: str,
    data_path: str,
    output_path: str,
    device: str,
    dlm_device: str = None,
    ar_device: str = None,
    max_samples: int = 30,
    ids_file: str = None,
    shard_idx: int = None,
    num_shards: int = None,
    methods: List[str] = None,
    resume: bool = True
):
    """
    AR Refinement 评估主函数

    Sharding 支持:
    - ids_file: 只跑指定 sample_id 的样本
    - shard_idx + num_shards: 按顺序切分 (e.g., --shard_idx 0 --num_shards 5 跑第一个 shard)
    - methods: 只跑指定方法 (ar_only, hybrid, dlm_only)

    断点续传:
    - resume: 如果为 True，会检查 checkpoint 文件并从断点继续
    - checkpoint 文件保存在 output_path + ".ckpt"
    """
    # 如果没有指定单独设备，使用默认 device
    dlm_device = dlm_device or device
    ar_device = ar_device or device

    # 默认方法：ar_only 和 hybrid（不跑 dlm_only 节省时间）
    if methods is None:
        methods = ["ar_only", "hybrid"]

    print(f"\n{'='*60}", flush=True)
    print(f"AR Refinement Evaluation", flush=True)
    print(f"Methods: {methods}", flush=True)
    if shard_idx is not None:
        print(f"Shard: {shard_idx}/{num_shards}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # 加载数据
    with open(data_path, 'r') as f:
        all_samples = [json.loads(line) for line in f]

    tool_list = get_canonical_tool_definitions()
    tool_name_sequence = [t["name"] for t in tool_list]
    valid_tools = set(tool_name_sequence)
    filtered_samples = filter_samples_by_tools(all_samples, valid_tools)
    if not filtered_samples:
        raise ValueError("No samples remain after filtering with canonical tool set.")
    print(f"Total canonical tools: {len(valid_tools)}", flush=True)
    print(f"Filtered samples: {len(filtered_samples)} / {len(all_samples)}", flush=True)

    # === 样本选择逻辑 ===
    import random
    random.seed(42)  # 可复现

    # 如果指定了 ids_file，只跑指定的样本
    if ids_file:
        with open(ids_file, 'r') as f:
            target_ids = set(line.strip() for line in f if line.strip())
        samples = [s for s in filtered_samples if s.get('id', '') in target_ids]
        print(f"Loaded {len(target_ids)} IDs from {ids_file}, matched {len(samples)} samples", flush=True)
    else:
        # 分层采样：按类型均匀采样
        dag_samples = [s for s in filtered_samples if s.get('type') == 'dag']
        chain_samples = [s for s in filtered_samples if s.get('type') == 'chain']
        single_samples = [s for s in filtered_samples if s.get('type') == 'single']

        # 每类采样 max_samples // 3
        per_type = max_samples // 3

        random.shuffle(dag_samples)
        random.shuffle(chain_samples)
        random.shuffle(single_samples)

        samples = (dag_samples[:per_type] +
                   chain_samples[:per_type] +
                   single_samples[:per_type])

    # 如果指定了 sharding，切分样本
    if shard_idx is not None and num_shards is not None:
        total = len(samples)
        shard_size = (total + num_shards - 1) // num_shards  # ceiling division
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        samples = samples[start:end]
        print(f"Shard {shard_idx}: samples [{start}:{end}] = {len(samples)} samples", flush=True)

    # 统计实际样本分布
    n_dag = sum(1 for s in samples if s.get('type') == 'dag')
    n_chain = sum(1 for s in samples if s.get('type') == 'chain')
    n_single = sum(1 for s in samples if s.get('type') == 'single')
    print(f"Evaluating {len(samples)} samples (DAG:{n_dag}, Chain:{n_chain}, Single:{n_single})", flush=True)

    # 初始化模型 (按需加载)
    print(f"DLM device: {dlm_device}, AR device: {ar_device}", flush=True)

    # 只加载需要的模型
    need_dlm = any(m in ["dlm_only", "hybrid"] for m in methods)
    need_ar = any(m in ["ar_only", "hybrid", "ar_two_stage"] for m in methods)

    dlm = DLMToolSelector(dlm_path, dlm_device) if need_dlm else None
    ar = AREdgeBuilder(ar_path, ar_device) if need_ar else None

    results = {m: [] for m in methods}

    # === 断点续传逻辑 ===
    checkpoint_path = output_path + ".ckpt"
    start_idx = 0
    completed_ids = set()

    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                ckpt_data = json.load(f)
            results = ckpt_data.get("results", {m: [] for m in methods})
            completed_ids = set(ckpt_data.get("completed_ids", []))
            print(f"[RESUME] Loaded checkpoint with {len(completed_ids)} completed samples", flush=True)
        except Exception as e:
            print(f"[RESUME] Failed to load checkpoint: {e}, starting fresh", flush=True)

    def save_checkpoint():
        """保存 checkpoint"""
        ckpt_data = {
            "completed_ids": list(completed_ids),
            "results": results,
            "methods": methods
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(ckpt_data, f, ensure_ascii=False)

    for i, sample in enumerate(samples):
        sample_id = sample.get('id', str(i))

        # 跳过已完成的样本
        if sample_id in completed_ids:
            continue

        instruction = get_sample_instruction(sample)
        gt_nodes = get_sample_task_nodes(sample)
        gt_tools = [n.get('task', '') for n in gt_nodes]
        gt_links = get_sample_task_links(sample)
        gt_edges = [(l.get('source', ''), l.get('target', '')) for l in gt_links]
        task_type = sample.get('type', 'unknown')

        full_prompt = build_edge_construction_prompt(instruction, tool_name_sequence, tool_list)
        n_completed = len(completed_ids)
        log_parts = [f"[{n_completed+1}/{len(samples)}] {task_type}:"]

        # === 方法 1: DLM 直接生成 ===
        if "dlm_only" in methods:
            dlm_output = dlm.select_tools(full_prompt, max_tokens=512)
            dlm_tools = extract_tools_from_text(dlm_output, valid_tools)
            dlm_edges = extract_edges_from_text(dlm_output)
            dlm_metrics = compute_metrics(dlm_tools, gt_tools, dlm_edges, gt_edges)
            log_parts.append(f"DLM F1={dlm_metrics['tool_f1']:.2f}/Edge={dlm_metrics['edge_recall']:.2f}")
            results["dlm_only"].append({
                "sample_id": sample_id,
                "task_type": task_type,
                "dlm_output": dlm_output,
                "pred_tools": dlm_tools,
                "gt_tools": gt_tools,
                **dlm_metrics
            })

        # === 方法 2: AR 直接生成（单阶段） ===
        if "ar_only" in methods:
            ar_output = ar.build_plan(full_prompt, max_tokens=512)
            ar_tools = extract_tools_from_text(ar_output, valid_tools)
            ar_edges = extract_edges_from_text(ar_output)
            ar_metrics = compute_metrics(ar_tools, gt_tools, ar_edges, gt_edges)
            log_parts.append(f"AR F1={ar_metrics['tool_f1']:.2f}/Edge={ar_metrics['edge_recall']:.2f}")
            results["ar_only"].append({
                "sample_id": sample_id,
                "task_type": task_type,
                "pred_tools": ar_tools,
                "gt_tools": gt_tools,
                **ar_metrics
            })

        # === 方法 3: AR-two-stage (AR 选工具 + AR 连边) ===
        if "ar_two_stage" in methods:
            # Step 1: AR 作为 proposer 选工具
            tool_prompt_ar = build_tool_selection_prompt(instruction, tool_list)
            ar_tool_output = ar.build_plan(tool_prompt_ar, max_tokens=128)
            ar_selected_tools = extract_tools_from_text(ar_tool_output, valid_tools)

            # Step 2: AR 在选出的工具上构建 plan
            if ar_selected_tools:
                edge_prompt_ar2 = build_edge_construction_prompt(instruction, ar_selected_tools, tool_list)
                ar2_output = ar.build_plan(edge_prompt_ar2, max_tokens=512)
                ar2_tools = extract_tools_from_text(ar2_output, valid_tools)
                ar2_edges = extract_edges_from_text(ar2_output)
            else:
                # fallback：如果 AR proposer 没选到工具，退回到单阶段行为
                ar2_output = ar.build_plan(full_prompt, max_tokens=512)
                ar2_tools = extract_tools_from_text(ar2_output, valid_tools)
                ar2_edges = extract_edges_from_text(ar2_output)

            ar2_metrics = compute_metrics(ar2_tools, gt_tools, ar2_edges, gt_edges)
            log_parts.append(f"AR2 F1={ar2_metrics['tool_f1']:.2f}/Edge={ar2_metrics['edge_recall']:.2f}")
            results["ar_two_stage"].append({
                "sample_id": sample_id,
                "task_type": task_type,
                "ar_selected_tools": ar_selected_tools,
                "pred_tools": ar2_tools,
                "gt_tools": gt_tools,
                **ar2_metrics
            })

        # === 方法 4: Hybrid (DLM 选工具 + AR 连边) ===
        if "hybrid" in methods:
            # Step 1: DLM 选工具
            tool_prompt = build_tool_selection_prompt(instruction, tool_list)
            dlm_tool_output = dlm.select_tools(tool_prompt, max_tokens=128)
            selected_tools = extract_tools_from_text(dlm_tool_output, valid_tools)

            # Step 2: AR 根据 DLM 选的工具构建 plan
            if selected_tools:
                edge_prompt = build_edge_construction_prompt(instruction, selected_tools, tool_list)
                hybrid_output = ar.build_plan(edge_prompt, max_tokens=512)
                hybrid_tools = extract_tools_from_text(hybrid_output, valid_tools)
                hybrid_edges = extract_edges_from_text(hybrid_output)
            else:
                # 如果 DLM 没选到工具，回退用所有工具
                hybrid_output = ar.build_plan(full_prompt, max_tokens=512)
                hybrid_tools = extract_tools_from_text(hybrid_output, valid_tools)
                hybrid_edges = extract_edges_from_text(hybrid_output)

            hybrid_metrics = compute_metrics(hybrid_tools, gt_tools, hybrid_edges, gt_edges)
            log_parts.append(f"Hybrid F1={hybrid_metrics['tool_f1']:.2f}/Edge={hybrid_metrics['edge_recall']:.2f}")
            results["hybrid"].append({
                "sample_id": sample_id,
                "task_type": task_type,
                "dlm_selected_tools": selected_tools,
                "pred_tools": hybrid_tools,
                "gt_tools": gt_tools,
                **hybrid_metrics
            })

        print(" | ".join(log_parts), flush=True)

        # 标记完成并保存 checkpoint
        completed_ids.add(sample_id)
        save_checkpoint()

    # 汇总
    summary = {}
    for method in methods:
        if results[method]:
            summary[method] = {
                "n_samples": len(results[method]),
                "avg_tool_f1": np.mean([r["tool_f1"] for r in results[method]]),
                "avg_edge_recall": np.mean([r["edge_recall"] for r in results[method]]),
            }

            # 按任务类型分组
            by_type = {}
            for task_type in ["single", "chain", "dag"]:
                type_results = [r for r in results[method] if r["task_type"] == task_type]
                if type_results:
                    by_type[task_type] = {
                        "n": len(type_results),
                        "tool_f1": np.mean([r["tool_f1"] for r in type_results]),
                        "edge_recall": np.mean([r["edge_recall"] for r in type_results])
                    }
            summary[method]["by_type"] = by_type

    # 保存元信息
    meta = {
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "ids_file": ids_file,
        "methods": methods,
        "n_samples": len(samples)
    }
    output_data = {"meta": meta, "summary": summary, "results": results}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 打印结果
    print(f"\n{'='*60}", flush=True)
    print(f"AR REFINEMENT EVALUATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n{'Method':<15} {'N':<8} {'Tool F1':<12} {'Edge Recall':<12}", flush=True)
    print("-" * 50, flush=True)
    for method in methods:
        if method in summary:
            s = summary[method]
            print(f"{method:<15} {s['n_samples']:<8} {s['avg_tool_f1']:<12.3f} {s['avg_edge_recall']:<12.3f}", flush=True)

    print(f"\n--- By Task Type ---", flush=True)
    for method in methods:
        if method in summary:
            print(f"\n{method}:", flush=True)
            for t, stats in summary[method].get("by_type", {}).items():
                print(f"  {t} (n={stats['n']}): Tool F1={stats['tool_f1']:.2f}, Edge={stats['edge_recall']:.2f}", flush=True)

    print(f"\nResults saved to: {output_path}", flush=True)

    # 完成后删除 checkpoint 文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint removed: {checkpoint_path}", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AR Refinement Evaluation with Sharding Support")
    parser.add_argument("--dlm_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--ar_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="data/taskbench/taskbench_hf_improved_flattened.jsonl")
    parser.add_argument("--output_path", default="results/ar_refinement/hybrid_eval.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dlm_device", default=None, help="Device for DLM (overrides --device)")
    parser.add_argument("--ar_device", default=None, help="Device for AR (overrides --device)")
    parser.add_argument("--max_samples", type=int, default=30, help="Max samples (used if no ids_file)")

    # Sharding 参数
    parser.add_argument("--ids_file", default=None, help="File with sample IDs to run (one per line)")
    parser.add_argument("--shard_idx", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--methods", nargs="+", default=["ar_only", "hybrid"],
                        choices=["dlm_only", "ar_only", "ar_two_stage", "hybrid"],
                        help="Methods to run (default: ar_only hybrid)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable checkpoint resume (start fresh)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # 支持双卡：如果指定了 dlm_device/ar_device 则使用，否则用 device
    dlm_device = args.dlm_device if args.dlm_device else args.device
    ar_device = args.ar_device if args.ar_device else args.device

    run_ar_refinement_eval(
        dlm_path=args.dlm_path,
        ar_path=args.ar_path,
        data_path=args.data_path,
        output_path=args.output_path,
        device=args.device,
        dlm_device=dlm_device,
        ar_device=ar_device,
        max_samples=args.max_samples,
        ids_file=args.ids_file,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        methods=args.methods,
        resume=not args.no_resume
    )
