# Contributing Guide

Thank you for considering a contribution to this repository.

## Scope of contributions

We welcome improvements in:
- simulation design and fairness scenarios,
- modeling robustness and diagnostics,
- statistical validation and reproducibility,
- documentation and reviewer-facing clarity,
- testing and CI reliability.

## How to contribute

1. **Open an issue** describing the proposed change (bug, enhancement, documentation, or question).
2. **Fork and branch** from the latest default branch.
3. **Implement small, focused commits** with clear messages.
4. **Submit a pull request** with motivation, implementation notes, and test evidence.

## Reporting issues

Please include:
- expected vs observed behavior,
- minimal reproduction steps,
- Python version and dependency context,
- relevant logs/tracebacks,
- whether results impact paper-facing outputs (figures/tables).

## Pull request guidance

PRs should include:
- concise problem statement,
- design/algorithmic rationale,
- files changed and why,
- validation evidence (`ruff`, `py_compile`, and/or tests),
- notes on whether outputs in `results/` were regenerated.

## Coding style and quality

- Follow existing project structure and naming conventions.
- Prefer explicit, readable code over clever shortcuts.
- Keep functions focused and document non-obvious statistical/modeling choices.
- Do not introduce broad refactors in bug-fix PRs unless necessary.

## Documentation expectations

If behavior or outputs change, update:
- `README.md` (user-facing workflow),
- `docs/PAPER_MAP.md` (paper-to-code traceability),
- docstrings and inline comments where interpretation could be ambiguous.

## Reproducibility expectations

Contributions should preserve or improve reproducibility:
- keep deterministic seeds where applicable,
- avoid hidden side effects in pipeline execution,
- document any change that affects generated figures/tables.

## Testing expectations

Before opening a PR, run at minimum:

```bash
ruff check .
python -m py_compile main.py app.py src/*.py
```

If you add functionality, include targeted tests or reproducible scripts demonstrating correctness.

## Respectful collaboration

By participating, you agree to follow [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Please keep reviews constructive, specific, and evidence-based.
