# Consolidation validation, 9 November 2026

The complete repository suite was run in WSL Ubuntu with Python 3.12 after the received face code/protocol were added: **417 passed, 7 skipped, 1 failed**. The existing failing test is `tests/test_real_cases.py::test_guard_refuses_changed_inputs_and_missing_amendment`: the frozen execution guard requires Python 3.11. No guard or scientific test was relaxed. Earlier upstream-only suite: 401 passed, 8 skipped, same failure.

The aggregate-only paper build succeeds with 8 content pages (10 total main pages) and 15 supplement pages. `verification.json` records exact PDF hashes, no overfull/missing-glyph/reference warnings, no identifying-text scan hits and an exact official-style match. These are file checks, not scientific reproduction or submission eligibility.

All manuscript evidence reads are pinned to a48523d, while the checkout retains the 33 newer fork commits. The face loader uses a repository-relative default. Runtime copying is local-only; every file is compared by SHA-256 before removal of its former source. `archive/syncthing-20261109/preservation_manifest.json` records recovery decisions for the 456 non-external files.
