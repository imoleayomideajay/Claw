# Release Checklist

Use this checklist before cutting a public release.

## 1) Versioning and changelog

- [ ] Bump version identifiers where applicable (`CITATION.cff`, release notes, tagged artifacts).
- [ ] Update/prepare changelog summary (features, fixes, breaking changes, known limitations).
- [ ] Confirm release title and semantic version tag format (e.g., `v0.2.0`).

## 2) Validation and tests

- [ ] Run linting and static checks.
- [ ] Run project tests and/or reproducible validation scripts.
- [ ] Confirm key pipeline entrypoints execute (`python main.py`, `streamlit run app.py`).
- [ ] Confirm no unresolved high-severity errors in logs.

## 3) Regenerate research artifacts

- [ ] Regenerate figures and tables used in documentation/paper.
- [ ] Verify outputs in `results/figures/` and `results/tables/` are current.
- [ ] Validate scenario-comparison outputs and IAS summary tables.
- [ ] Spot-check representative outputs for numerical sanity.

## 4) Documentation consistency

- [ ] Ensure README reflects current workflow, scenarios, and outputs.
- [ ] Ensure `docs/PAPER_MAP.md` is synchronized with module responsibilities.
- [ ] Ensure `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` links are valid.
- [ ] Ensure any API/CLI behavior changes are documented.

## 5) Citation and metadata

- [ ] Update `CITATION.cff` version and `date-released`.
- [ ] Verify author list and repository URL are correct.
- [ ] Confirm preferred citation text aligns with current manuscript/preprint status.

## 6) Archival and DOI workflow (Zenodo or equivalent)

- [ ] Ensure GitHub–Zenodo integration is enabled.
- [ ] Create/tag release candidate commit.
- [ ] Trigger Zenodo archive capture for release tag.
- [ ] Verify DOI minted and add DOI badge/link in README.

## 7) GitHub release steps

- [ ] Push release tag.
- [ ] Draft GitHub Release notes (highlights + migration notes if needed).
- [ ] Attach relevant artifacts (if policy allows) or link to archived outputs.
- [ ] Mark pre-release vs stable appropriately.

## 8) Streamlit deployment check

- [ ] Confirm deployment environment uses compatible dependency versions.
- [ ] Run health-check path and verify app loads without runtime errors.
- [ ] Validate representative scenario run in deployed app.
- [ ] Verify user-facing messaging for fallback/model warnings is understandable.

## 9) Post-release

- [ ] Announce release in project channels.
- [ ] Open follow-up issues for deferred work.
- [ ] Record release provenance (commit hash, tag, DOI) for paper appendix/reproducibility log.
