import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tool_catalog import CANONICAL_TOOL_NAMES, filter_samples_by_tools  # noqa: E402
from utils import get_sample_instruction, get_sample_task_links, get_sample_task_nodes  # noqa: E402


def test_reconstructed_taskbench_counts_and_schema() -> None:
    path = ROOT / "data/taskbench/taskbench_hf_improved_flattened.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 7458
    assert Counter(row["type"] for row in rows) == {
        "single": 3067,
        "chain": 3596,
        "dag": 795,
    }
    first = rows[0]
    assert get_sample_instruction(first)
    assert get_sample_task_nodes(first)
    assert isinstance(get_sample_task_links(first), list)

    filtered = filter_samples_by_tools(rows)
    assert len(CANONICAL_TOOL_NAMES) == 23
    assert len(filtered) == 6496
    assert Counter(row["type"] for row in filtered) == {
        "single": 2915,
        "chain": 2936,
        "dag": 645,
    }


def test_released_split_sizes() -> None:
    ids = (ROOT / "data/ids_500.txt").read_text().splitlines()
    test_ids = (ROOT / "data/splits/ijcai_test_ids.txt").read_text().splitlines()
    train_ids = (
        ROOT / "data/splits/ijcai_train_ids_chain450_dag450_seed42.txt"
    ).read_text().splitlines()
    assert len(ids) == len(set(ids)) == 501
    assert len(test_ids) == len(set(test_ids)) == 501
    assert set(ids) == set(test_ids)
    assert len(train_ids) == len(set(train_ids)) == 900
    assert set(train_ids).isdisjoint(test_ids)
