import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_released_selector_reproduces_paper_values(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_selection_on_candidates.py"),
            "--candidates_path",
            str(ROOT / "artifacts/candidate_pools/taskbench_dream_k5_test334.json"),
            "--model_path",
            str(ROOT / "artifacts/value_function/plan_scorer_combo07_toolset.pkl"),
            "--out_json",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
    expected = {
        "k1_tool_f1": 0.6845855907233153,
        "k1_edge_recall": 0.24381950384944395,
        "heur_tool_f1": 0.6900471451369654,
        "heur_edge_recall": 0.3110849729113202,
        "scorer_tool_f1": 0.716250150381887,
        "scorer_edge_recall": 0.3092386655260907,
        "oracle_tool_f1": 0.7682874178383161,
        "oracle_edge_recall": 0.2829840319361277,
        "n_samples": 334.0,
    }
    assert summary == pytest.approx(expected, abs=1e-12)


def test_training_and_test_candidate_ids_are_disjoint() -> None:
    train = json.loads(
        (ROOT / "artifacts/candidate_pools/taskbench_dream_k5_train670.json").read_text()
    )
    test = json.loads(
        (ROOT / "artifacts/candidate_pools/taskbench_dream_k5_test334.json").read_text()
    )
    train_ids = {str(row["sample_id"]) for row in train["candidates"]}
    test_ids = {str(row["sample_id"]) for row in test["candidates"]}
    assert len(train_ids) == 670
    assert len(test_ids) == 334
    assert train_ids.isdisjoint(test_ids)
