# core/ — shared native decoder

One C++17 library (no OpenCV, no threads, deterministic) for the decoder hot path. Used from Python via pybind11 (research/eval)
and, in phase 2, from Android via the NDK/JNI. Replaces the hand-maintained Python/Kotlin duplicate.

## Pipeline (`core/src`)
1. `image.cpp` darkness normalisation (Otsu, paper/ink medians; uint8 or float32 input).
2. `register.cpp` coarse pose: moments (scale, centroid; principal axis for lines) + 36-angle NCC sweep at ~2.5 px/mm with
   parabolic peak interpolation (square blocks); then **edge-based homography fit** (no ECC): the reference contour (asset points +
   normals) is sampled along its normals in a wide window (area method, same measurement as the decoder), the offsets are fitted by an
   8-parameter homography update (Gauss-Newton, Tukey IRLS), window narrowed 1.2 -> 0.6 -> 0.3 mm. Validity of a sample for a wide
   window is the clean-single-edge test on the 20 px/mm reference raster.
3. `decode.cpp` nuisance fit (3 iterations: homography affine update + poly-4 Legendre warp + dot gain, normal equations + Cholesky,
   2-pass 4-sd outlier removal by Gram downdate), per-contour high-pass (4 chips), 3-MAD clip, matched filter, pilot z/gain with the
   high-pass-loss calibration, despread, exact soft-ML Golay(24,12), CRC-16. `accept = crc_ok && z > z_accept`. Sheet 1 (`rep_crc8`)
   uses the same code path with the additive-pilot statistic and CRC-8.
4. `api.cpp` `decode`, `register_block`, `decode_multi` (K crops: each registered to the reference, resampled into crop 0's frame,
   averaged, then decoded).

## Asset
`tools/export_android_ref.py` output (`<name>.json` + `<name>.bin`, unchanged format; it now also accepts the sheet-1 `rep_crc8` key).
`stegocore.asset_for(key)` exports on first use and caches under `core/build_assets/` (git-ignored). Existing: Android's
`android/app/src/main/assets/sheet2_ref.*` loads as is.

## Build
- Mac (CLI + Python module): `uv pip install --python .venv/bin/python pybind11 && core/build.sh`
  (the module lands in `core/python/`; `import sys; sys.path.insert(0, "core/python"); import stegocore`).
- Android arm64-v8a CLI: `sdkmanager --install "ndk;29.0.14206865" && core/build_android.sh` -> `core/build-android/stego_core_cli`.
- CLI: `stego_core_cli <asset.json> <crop.pgm> [more crops = multi-frame] [--reps N]` (prints per-stage ms).

## Python
```python
from stegocore import asset_for, decode, decode_multi, register
a = asset_for(model.key)                  # or geostego.decode.decode(img, model, backend="cpp")
r = decode(a, crop)                       # crop: 2-D uint8/float32; optional H (3x3 ref mm -> crop px) skips registration
r = decode_multi(a, [crop0, crop1, ...])  # roughly aligned crops of one block (same box in each frame, +- ~1 mm)
```
Result dict: hex, crc_ok, accept, z_pilot, amp_um, soft, bits, H, reg_score, timing_ms{darkness,coarse,edge,measure,fit,tail,golay,total}.
Parameters (`Params`, kwargs): `n_iter`, `iter_stride`, `fit_stride`, `step`, `reg_W/reg_iters/reg_step/reg_stride`, ...

## Tools / validation
`tools/eval_core.py` (26 conditions x 5 blocks, `--combine core|py-align|single`, `--cache DIR` to reuse crops), `tools/core_capture_range.py`,
`tools/eval_core_sheet1.py`. `STEGO_DEBUG=1` prints registration internals.
