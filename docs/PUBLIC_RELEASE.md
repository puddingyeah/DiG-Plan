# Public release boundary

This repository was assembled from a private research workspace using an explicit allowlist. The historical workspace and its Git history are not publication inputs.

## Included

- DiG-Plan inference, candidate collection, value-function, analysis, and controlled-study source code
- paper-facing TaskBench and API-Bank split IDs
- the compact TaskBench reconstruction described in `docs/DATA.md`
- two compact candidate pools, the exact value-function bundle, and its evaluation result
- paper figures used in the README
- environment pins, tests, citation metadata, licenses, and third-party notices

## Excluded

- migration archives, manifests, backups, and unrelated historical repository contents
- paper source, submission packages, rebuttal material, copyright forms, and correspondence
- internal reports, notes, proposals, experiment queues, job scripts, logs, PIDs, sentinels, and temporary files
- model checkpoints, pretrained weights, tokenizer/model caches, absolute symlinks, and partial large-file downloads
- raw hyperparameter sweeps, redundant candidate shards, benchmark clones, and generated build outputs
- credentials, environment files, host-specific absolute paths, and personally identifying operational metadata

These exclusions are enforced by `.gitignore` and `scripts/audit_public_release.py`. The audit intentionally permits only the released scikit-learn pickle and rejects common neural-model weight formats.

## Maintainer pre-push checklist

Run the following from a clean clone or worktree:

```bash
python -m pytest -q
python scripts/audit_public_release.py
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

Then verify that every staged file belongs to the inclusion list above. Never use a broad force-add command to bypass `.gitignore`. Publishing is a separate maintainer action; preparing this directory does not push or modify the remote repository.
