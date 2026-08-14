import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_audit_passes() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_public_release.py")],
        check=True,
        cwd=ROOT,
    )
