# CI Pipeline + Provenance Enforcement

## Status: ✅ Fully Done

## What It Does

Every push triggers:
1. `pytest` — runs all 15 test files
2. `python scripts/make_tables.py --check` — verifies README tables match committed artifacts

If any table row doesn't trace to a real JSON artifact → CI **fails**. Prevents fabricated
numbers from ever appearing in the README again.

## Anti-Fabrication Guards

- `tests/test_provenance.py` — tables match artifacts + bans specific fabricated phrases
- `test_no_fabricated_numbers_resurface()` — blocks previously retracted claims
- Retraction documented in README lines 17–22

## Key Files

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | CI workflow |
| `tests/test_provenance.py` | Provenance enforcement |
| `scripts/make_tables.py` | Autogenerate + verify tables |

## Context

This was created after discovering that earlier versions of the repo contained fabricated
results (66%/131% degradation claim, unmatched benchmark table, divide-by-10 correction).
The CI pipeline is a structural guard against recurrence.
