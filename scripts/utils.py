"""
Utility functions for DiG-Plan experiments.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any


def load_taskbench_data(data_dir: str, n_tools_filter: List[int] = None) -> List[Dict]:
    """Load TaskBench data from jsonlines file"""
    data_path = os.path.join(data_dir, "data.json")
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            if n_tools_filter is None or item["n_tools"] in n_tools_filter:
                data.append(item)
    return data


def load_tool_descriptions(data_dir: str) -> Dict[str, Any]:
    """Load tool descriptions"""
    tool_path = os.path.join(data_dir, "tool_desc.json")
    with open(tool_path) as f:
        return json.load(f)


def get_sample_instruction(sample: Dict[str, Any]) -> str:
    """
    Return the natural-language request field from a TaskBench-style sample.

    Historical local copies in this repo often used `instruction`, while the
    official TaskBench release uses `user_request`.
    """
    for key in ("instruction", "user_request", "request", "query"):
        val = sample.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _coerce_json_list(value: Any) -> List[Any]:
    """Normalize TaskBench-style list fields stored as JSON strings or Python lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return []
        return obj if isinstance(obj, list) else []
    return []


def get_sample_task_nodes(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return GT nodes from either official (`task_nodes`) or legacy (`tool_nodes`) fields."""
    for key in ("task_nodes", "tool_nodes", "nodes"):
        if key in sample:
            return _coerce_json_list(sample.get(key))
    return []


def get_sample_task_links(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return GT links from either official (`task_links`) or legacy (`tool_links`) fields."""
    for key in ("task_links", "tool_links", "links"):
        if key in sample:
            return _coerce_json_list(sample.get(key))
    return []


def format_tool_prompt(instruction: str, tools: List[Dict]) -> str:
    """Format prompt for tool-use task"""
    tool_str = ""
    for tool in tools:
        if "input-type" in tool:
            # Type-specific format
            tool_str += f"- {tool['task']}: Input: {tool['input-type']}, Output: {tool['output-type']}\n"
        else:
            # API format
            params = ", ".join([f"{p['name']}: {p['type']}" for p in tool.get('parameters', [])])
            tool_str += f"- {tool['id']}: {tool['desc']} | Parameters: ({params})\n"

    prompt = f"""You are a helpful assistant that can use tools to complete tasks.

Available tools:
{tool_str}

Task: {instruction}

Generate a step-by-step plan with tool calls in the following JSON format:
{{
  "steps": [
    {{"step": 1, "tool": "ToolName", "arguments": ["arg1", "arg2"], "output_var": "$v1"}},
    ...
  ]
}}

Plan:"""
    return prompt


def parse_tool_output(output: str) -> Dict:
    """Parse model output to extract tool calls"""
    try:
        # Try to extract JSON from output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start != -1 and end > start:
            json_str = output[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return {"steps": [], "parse_error": True}


def compute_metrics(pred_nodes: List[str], gold_nodes: List[str],
                   pred_links: List[tuple], gold_links: List[tuple]) -> Dict[str, float]:
    """Compute evaluation metrics"""
    # Node F1
    pred_set = set(pred_nodes)
    gold_set = set(gold_nodes)

    if len(pred_set) == 0 and len(gold_set) == 0:
        node_precision = node_recall = node_f1 = 1.0
    elif len(pred_set) == 0:
        node_precision = node_recall = node_f1 = 0.0
    elif len(gold_set) == 0:
        node_precision = node_recall = node_f1 = 0.0
    else:
        correct = len(pred_set & gold_set)
        node_precision = correct / len(pred_set)
        node_recall = correct / len(gold_set)
        node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall) if (node_precision + node_recall) > 0 else 0.0

    # Edge F1
    pred_edge_set = set(pred_links)
    gold_edge_set = set(gold_links)

    if len(pred_edge_set) == 0 and len(gold_edge_set) == 0:
        edge_precision = edge_recall = edge_f1 = 1.0
    elif len(pred_edge_set) == 0:
        edge_precision = edge_recall = edge_f1 = 0.0
    elif len(gold_edge_set) == 0:
        edge_precision = edge_recall = edge_f1 = 0.0
    else:
        correct = len(pred_edge_set & gold_edge_set)
        edge_precision = correct / len(pred_edge_set)
        edge_recall = correct / len(gold_edge_set)
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0.0

    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
    }


def get_experiment_id() -> str:
    """Generate experiment ID based on timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage"""
    # Keep data/metric helpers importable in the CPU-only analysis environment.
    # Torch is only required by inference and controlled-study entry points.
    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {}

    result = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        result[f"gpu_{i}_allocated_gb"] = allocated
        result[f"gpu_{i}_reserved_gb"] = reserved
    return result


def save_experiment_results(results: Dict, output_dir: str, exp_name: str):
    """Save experiment results to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{exp_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
