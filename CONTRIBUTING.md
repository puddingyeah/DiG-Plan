# Contributing

Bug reports, reproduction reports, and focused pull requests are welcome.

1. Create a virtual environment and install `requirements-analysis.txt` for CPU-only work or `requirements-inference.txt` for model inference.
2. Run `python -m pytest -q`.
3. Run `python scripts/audit_public_release.py` before committing.
4. Do not commit downloaded model weights, raw private experiment outputs, credentials, absolute machine paths, or files excluded by `.gitignore`.

Please describe the benchmark split, model revision, seed, hardware, and exact command when reporting a result difference.
