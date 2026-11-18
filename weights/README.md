# Included Dense adapters

This directory contains the completed Phase-2 adapter for M1, M2, and M3.
The `.pt` files are already inference-only BF16 checkpoints and are tracked
with Git LFS. Run `git lfs pull` after cloning.

`manifest.json` records model directions, parameter counts, byte sizes,
source runs, and SHA-256 checksums. Phase-1 and optimizer checkpoints are not
included because Phase-2 is the final adapter used for evaluation/inference.
