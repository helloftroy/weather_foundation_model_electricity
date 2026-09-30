# Candidate Prithvi WxC embedding tensors

**Status: confirmed from source** (NASA-IMPACT/Prithvi-WxC `main` branch --
the repo's default, and what `git clone` with no `-b` flag actually checks
out; corrected 2026-09-30 after initial research against `develop`
surfaced a real signature mismatch on the cluster -- `develop` and `main`
have diverged: `main` additionally requires `mask_ratio_targets` at
construction, raising `NotImplementedError` if `>0.0`, and its current own
demo notebook defaults to `masking_mode="global"`, `encoder_shifting=True`,
`decoder_shifting=True`, `masking_ratio=0.0`, which `src/prithvi/
extract_embeddings.py` and `src/prithvi/sample_validation.py` now match).
`PrithviWxC/model.py` + `PrithviWxC/dataloaders/merra2.py`, and the model's
`config.yaml`, fetched 2026-09-29/30. No cluster access was needed for this
-- it's a public repo -- see `src/prithvi/extract_embeddings.py` for the
implementation. `merra2.py`'s file-reading logic (the load-bearing parts
`build_daily_prithvi_files.py`/`crop_climatology.py` depend on) is
unchanged between the two branches -- re-diffed to confirm.

## Architecture summary

Encoder-decoder ViT with local+global ("Hiera x MaxViT") attention over a
windowed token layout. Input `[batch, time=2, parameter=160, lat, lon]` is
patch-embedded (`patch_size_px=[2,2]`), then grouped into "mask units"
(`mask_unit_size_px=[30,32]` pixels each). Tokens are reshaped to
`[batch, global_seq, local_seq, embed_dim]` where `global_seq` indexes mask
units and `local_seq` indexes patches within a mask unit.

Pretrained `prithvi.wxc.2300m.v1` config:
```
embed_dim: 2560
n_blocks_encoder: 12
n_blocks_decoder: 2
patch_size_px: [2, 2]
mask_unit_size_px: [30, 32]     # -> 15 deg lat x 20 deg lon at native MERRA-2 spacing
n_lats_px: 360, n_lons_px: 576  # the GLOBAL grid used at pretraining time
```

## The extraction point

`PrithviWxC.forward()`:
```python
unmasked = <tokens after masking, or all tokens if mask_ratio_inputs=0>
x_encoded = self.encoder(unmasked)   # <-- extraction point
...
x_decoded = self.decoder(recon)      # decoder/output: NOT needed for embeddings
```

With `mask_ratio_inputs=0.0`, nothing is masked, so `x_encoded` has shape
`[batch, n_global_mu, n_local_mu, embed_dim]` -- **every** spatial token,
fully structured, no forced pooling. Captured via a forward hook on
`model.encoder` (`register_forward_hook`) rather than reimplementing
`forward()`.

Two things this requires, confirmed necessary (not optional conveniences):
- `residual="climate"` -- the pretrained `patch_embedding_static` layer's
  weight shape (`in_channels + in_channels_static` input channels) was
  trained with this mode; switching to `"none"`/`"temporal"` would produce a
  shape mismatch on `load_state_dict(strict=True)`. This means the
  climatology data IS needed even though we discard the decoder's output.
- `positional_encoding="fourier"` -- encodes lat/lon live from the actual
  static-tensor values, not a fixed-size lookup table. This is *why*
  regional cropping is safe at all (see below) -- no learned weight is
  shaped by `n_lats_px`/`n_lons_px`.

## Regional cropping is architecturally valid

None of `PrithviWxC`'s weight tensors are shaped by `n_lats_px`/`n_lons_px`
directly: `patch_embedding`/`patch_embedding_static` are ordinary
spatially-agnostic conv layers, and Fourier positional encoding is computed
per-call from real lat/lon values. `n_lats_px`/`n_lons_px` only drive
`global_shape_mu`/`local_shape_mu`, used purely for reshaping and mask-index
bookkeeping. So a regional crop loads the pretrained `state_dict` cleanly, as
long as `n_lats_px % mask_unit_size_px[0] == 0` and likewise for lon (the
model's own constructor asserts this).

This matters practically, not just architecturally: running the model on the
full global 360x576 grid with ~0 masking is not feasible for a full year on
modest GPU resources -- the official demo notebook needs `mask_ratio=0.99`
just to stay small. Our regional crop (60x64 px, see
`docs/merra2_region_and_variables.md`) is ~54x fewer pixels, which is what
makes a full-year extraction affordable.

Also confirmed: `Merra2Dataset`'s `upper_shape`/`surface_shape` properties
hardcode a global `(361, 576)` shape, but grep shows they're **dead code** --
never referenced by the actual file-reading path, which dynamically sizes
everything from each file's own `lat`/`lon` dataset lengths. So
`Merra2Dataset` and `preproc()` are used completely unmodified on our
regional data; we only had to build daily merged files in its expected
format (see `src/merra2/build_daily_prithvi_files.py`).

## Compact representations saved (both, per the "save both if cheap" guidance)

For our chosen 2x2-mask-unit crop (`global_shape_mu = (2, 2)`, 4 regions,
`embed_dim=2560`):

- **`global_emb_*`** (2560 cols): `x_encoded.mean(dim=(1,2))` -- mean over
  every token. Maximally compact; loses all regional structure.
- **`region{0..3}_emb_*`** (4 x 2560 = 10240 cols): `x_encoded.mean(dim=2)`
  -- mean over local tokens only, keeping one vector per mask unit (per
  quadrant of the regional box). Falls directly out of the model's own
  token structure -- no extra pooling grid invented. Geographic center of
  each region is written alongside the embeddings parquet (`region_index`,
  `lat_center`, `lon_center`) by `extract_embeddings.py`.

Both are saved by default; `local/train_catboost_comparisons.py` uses only
`global_emb_*` for the first pass (simplest, per "start simple"), with the
region-level columns available for a later, more spatially-resolved
comparison without needing to re-run the GPU extraction.

## input_time / lead_time choice

Fixed at `input_time=-6`, `lead_time=0` for every timestamp -- an explicitly
supported pretraining regime ("0-hour ahead" nowcasting, per the model
card), valid because `-6` is in the pretrained input-delta set
`[-3,-6,-9,-12]`. This means each timestamp's embedding depends only on data
up to and including that timestamp (T and T-6h) -- no future leakage into a
downstream demand model. Held fixed across the whole year so embeddings are
comparable.

## Remaining validate-on-cluster items

- [ ] Confirm `configs/merra2_variables.yaml`'s GES-DISC-collection mapping
      is right (variable *names* are certain, from the source; which
      *collection* each lives in is standard-schema best-effort, unverified
      against an actual download).
- [ ] Confirm the tavg1-vs-instantaneous timestamp reconciliation in
      `build_daily_prithvi_files.py` (nearest-neighbor resample to the
      synoptic 3-hour grid) doesn't silently misalign anything -- spot-check
      a few merged daily files against the raw GES DISC downloads.
- [ ] Run `extract_embeddings.py` against the small MERRA-2 sample (Phase 1
      step 3's 2-day pull) before the full year, and sanity-check
      `global_shape_mu`/`embed_dim` printed at startup match the numbers
      above.
