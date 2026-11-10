# Recovered Syncthing history

This directory preserves unique historical files recovered before removal of a non-Git Syncthing copy. It is an archive, not the active manuscript or current experiment implementation. The active paper is `paper/cvpr2027/`. Received frozen face code, protocol and aggregate outputs use their canonical paths.

`preservation_manifest.json` records all 456 non-external files, including exact copies and LF-only duplicates recoverable from Git. Differing historical versions were retained without overwriting current sources. Temporary build renderings and downloaded templates are retained in the preservation commit, then removed from the final working tree. Git history is the recovery mechanism.

Publication-language exceptions are immutable historical evidence: the three BRATS literature notes and the old metrics-board source/report retain their original Russian passages. New documentation, manuscript text, figure labels and commit messages are English. These notes are not verified citations or current scientific conclusions. Do not inherit archived AGENTS instructions outside this archive.

Third-party code, virtual environments and their images/weights in `external/` are not public evidence. They are retained once in the canonical clone's ignored runtime directory, with a local hash-verification inventory.
