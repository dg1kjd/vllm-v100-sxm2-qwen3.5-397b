# Security Policy

> This replaces upstream vLLM's SECURITY.md, which describes *their*
> vulnerability-management process and does not apply to this fork.

## What this repository is (and what that means for security posture)

This is a **pinned deployment-recipe snapshot** of a vLLM fork for a specific
target (8× Tesla V100-SXM2, Qwen3.5-397B AWQ) — not a maintained
general-purpose library. The security posture follows from that, deliberately:

- **Dependency CVEs are handled upstream.** The full vLLM requirements tree
  (test/dev/optional extras) is never installed by this recipe; report and
  track library-dependency vulnerabilities against
  [vllm-project/vllm](https://github.com/vllm-project/vllm).
- **The supported install path is the curated venv** described in the README
  (`--no-deps` engine install + explicitly pinned serving dependencies). Pins
  are bumped **deliberately, with the rig regression suite** (`tests/fleet/`)
  — which is why Dependabot version-update PRs are disabled here. A past
  example of why blind bumps are dangerous is in the README gotchas
  (prometheus instrumentator × starlette: 500 on every request).
- **Code scanning (CodeQL) is not enabled**: it would report against ~2M lines
  of inherited upstream code that this fork does not patch independently.
  Quality assurance for the recipe itself is the fleet regression suite and
  the validated-configuration claims in the README.
- The serving endpoint this recipe produces is **not hardened for hostile
  networks** (no auth by default). Run it on a trusted network or behind your
  own reverse proxy / auth layer. See also the
  [vLLM security guide](https://docs.vllm.ai/en/latest/usage/security.html)
  and [PyTorch's security policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
  for model-handling recommendations — both still apply.

## Reporting a vulnerability

- For issues in **this fork's own additions** (SM70 patches, deploy scripts,
  SwiReasoning/vision changes, docs): use GitHub's **private vulnerability
  reporting** on this repository (Security tab → "Report a vulnerability"), or
  a regular issue if the problem is not sensitive.
- For issues in **upstream vLLM or its dependencies**: use
  [upstream's vulnerability submission form](https://github.com/vllm-project/vllm/security/advisories/new).

Best-effort response by a single maintainer; no SLA is promised (see the
README's no-warranty disclaimer).
