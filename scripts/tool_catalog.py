"""
Utility helpers for the canonical TaskBench tool library (23 tools).
All evaluation scripts should import from here instead of hardcoding
tool names or truncating the list in prompts.
"""

from __future__ import annotations

from typing import List, Dict, Iterable, Set, Any, Optional
import json
from utils import get_sample_task_nodes

# Ordered canonical tool list used throughout the paper.
CANONICAL_TOOL_NAMES: List[str] = [
    "Automatic Speech Recognition",
    "Text-to-Speech",
    "Visual Question Answering",
    "Sentence Similarity",
    "Depth Estimation",
    "Image Classification",
    "Token Classification",
    "Object Detection",
    "Image Editing",
    "Audio-to-Audio",
    "Conversational",
    "Translation",
    "Document Question Answering",
    "Image Segmentation",
    "Summarization",
    "Text Generation",
    "Question Answering",
    "Tabular Classification",
    "Audio Classification",
    "Text-to-Image",
    "Image-to-Image",
    "Text-to-Video",
    "Image-to-Text",
]

# Minimal English descriptions so prompts have consistent semantics.
TOOL_DESCRIPTIONS: Dict[str, str] = {
    "Automatic Speech Recognition": "Transcribe speech audio into text.",
    "Text-to-Speech": "Convert text into natural-sounding audio.",
    "Visual Question Answering": "Answer questions about an image and text context.",
    "Sentence Similarity": "Measure semantic similarity between two text inputs.",
    "Depth Estimation": "Predict a depth map from an RGB image.",
    "Image Classification": "Classify an image into predefined categories.",
    "Token Classification": "Tag each text token (e.g., NER, POS).",
    "Object Detection": "Detect objects in an image with bounding boxes.",
    "Image Editing": "Edit or transform an image according to instructions.",
    "Audio-to-Audio": "Enhance or transform an audio clip.",
    "Conversational": "Hold a multi-turn natural language conversation.",
    "Translation": "Translate text from one language to another.",
    "Document Question Answering": "Answer questions based on document content.",
    "Image Segmentation": "Segment an image into pixel-level regions.",
    "Summarization": "Summarize long-form text into concise highlights.",
    "Text Generation": "Generate or continue text passages.",
    "Question Answering": "Answer questions based on textual context.",
    "Tabular Classification": "Classify tabular data rows.",
    "Audio Classification": "Classify an audio clip into categories.",
    "Text-to-Image": "Generate an image from a text description.",
    "Image-to-Image": "Transform an image into another style or variant.",
    "Text-to-Video": "Generate a short video from a text description.",
    "Image-to-Text": "Produce textual captions for images.",
}


def get_canonical_tool_definitions() -> List[Dict[str, str]]:
    """Return the canonical tool list with descriptions."""
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS.get(name, "No description available."),
        }
        for name in CANONICAL_TOOL_NAMES
    ]


def load_taskbench_tool_definitions(tool_desc_path: str) -> List[Dict[str, str]]:
    """
    Load tool definitions from a TaskBench domain folder's `tool_desc.json`.

    TaskBench's `tool_desc.json` is typically a dict with a `nodes` field, where each node has:
      - `id`: tool name
      - `desc`: description
    Some domains (e.g., dailylifeapis) may include parameter schemas instead of I/O types.
    """
    with open(tool_desc_path, "r", encoding="utf-8") as f:
        obj: Any = json.load(f)
    nodes: Any = obj.get("nodes") if isinstance(obj, dict) else obj
    if not isinstance(nodes, list):
        raise ValueError(f"Unexpected tool_desc format (expected list or dict with 'nodes'): {tool_desc_path}")

    out: List[Dict[str, str]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        name = n.get("id") or n.get("name") or n.get("task")
        if not name:
            continue
        desc = n.get("desc") or n.get("description") or ""
        params = n.get("parameters")
        if isinstance(params, list):
            pnames = [
                p.get("name")
                for p in params
                if isinstance(p, dict) and p.get("name")
            ]
            if pnames:
                desc = (desc + " " if desc else "") + f"Parameters: {', '.join(pnames)}."
        out.append({"name": str(name), "description": str(desc) if desc else "No description available."})

    if not out:
        raise ValueError(f"No tool definitions loaded from: {tool_desc_path}")
    return out


def get_tool_definitions(tool_desc_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Return tool definitions from a TaskBench `tool_desc.json`, or the canonical 23-tool library if None."""
    return load_taskbench_tool_definitions(tool_desc_path) if tool_desc_path else get_canonical_tool_definitions()


def filter_samples_by_tools(
    samples: Iterable[Dict],
    allowed_tools: Iterable[str] = None,
) -> List[Dict]:
    """
    Keep only TaskBench samples whose ground-truth tools are inside the allowed set.
    This prevents us from evaluating on tasks that require tools outside the 23-tool
    library (which the planner cannot possibly call).
    """
    allowed: Set[str] = (
        set(allowed_tools) if allowed_tools is not None else set(CANONICAL_TOOL_NAMES)
    )
    filtered = []
    for sample in samples:
        nodes = get_sample_task_nodes(sample)
        # Handle both dict nodes and string nodes
        gt_tools = set()
        for node in nodes:
            if isinstance(node, dict):
                task = node.get("task", "")
                if task:
                    gt_tools.add(task)
            elif isinstance(node, str) and node:
                gt_tools.add(node)
        if gt_tools and gt_tools.issubset(allowed):
            filtered.append(sample)
    return filtered
