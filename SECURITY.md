# Security policy

DiG-Plan does not require API keys for the released open-model pipeline. Never commit Hugging Face tokens or other credentials.

The released value function is a Python pickle. Pickles can execute code while loading; only load the repository copy after verifying its checksum, or a model you trained yourself. Do not accept untrusted pickle files through services built on this code.

Please report a security issue through GitHub's private security-advisory feature rather than a public issue.
