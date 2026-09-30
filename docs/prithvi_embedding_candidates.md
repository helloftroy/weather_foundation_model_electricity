# Candidate Prithvi WxC embedding tensors

**Status: not yet filled in.** Requires cluster access to clone the repo and
either read the model source or run `src/prithvi/inspect_architecture.py`
against a loaded checkpoint (Phase 1 step 2).

From the paper (arXiv:2409.13598) and repo README, known so far:

- Architecture: encoder-decoder, "scalable 2D vision transformer" with local
  and global attention over a windowed token layout `(batch, windows, tokens,
  features)`.
- 2.3B parameters, trained on 160 MERRA-2 variables.
- Takes two input timestamps, produces one (possibly future) output timestamp.

## To determine on the cluster

- [ ] Exact encoder output shape and where encoder ends / decoder begins in
      the module tree (`model.named_modules()`).
- [ ] Whether encoder tokens retain spatial (lat/lon) structure, or are
      flattened/pooled before the decoder.
- [ ] Embedding dimension per token, and total token count for our regional
      crop size.
- [ ] Whether a global-pooled vector, a per-location token grid, or both are
      more appropriate to feed to CatBoost (a per-zone electricity model
      likely wants some spatial resolution retained, not just one global
      vector -- but confirm token layout before deciding).
- [ ] Whether masking (50% ratio used in pretraining) needs to be disabled
      or handled specially for representation extraction vs. pretraining.

## Recorded findings (fill in after inspection)

(empty)
