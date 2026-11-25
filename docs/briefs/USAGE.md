# Copilot CLI usage per task

Measured with `copilot -p ... --usage-output-file`. Settings: `model: auto`, `contextTier: default`, memory off.

| Task | Model picked by auto | Premium requests | AI credits | Input tokens (cached) | Output tokens | Wall time | Review notes |
|---|---|---|---|---|---|---|---|
| P1-T1 | mai-code-2-flash | 1 | 1.03 | 308.2k (253.4k) | 4.5k | 1m 37s | Tests passed first try. Reviewer fixes: removed `exclude = ["tests"]` and `E501` ignore it added to ruff config, restored `uv.lock` it deleted, replaced an empty package docstring. |
| P1-T2 | mai-code-2-flash | 1 | 0.89 | 151.1k (82.9k) | 2.4k | 52s | Tests passed first try, linear-time contingency implementation, no reviewer code changes. Reviewer reformatted the Python block inside the brief, which `ruff format` also checks. |
