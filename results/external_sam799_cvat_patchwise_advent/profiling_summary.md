# Profiling Summary

Patchwise runtime and memory summary on SAM799-CVAT.

- `runtime_total_sec`: image load + patchwise inference/stitching + threshold/postprocess/metrics; excludes optional asset writes
- `predict_total_sec`: patch preprocessing + batched forward passes + resize-back + probability averaging
- `global_patches_per_sec` and `global_megapixels_per_sec` are computed from aggregate predict time
- GPU peaks are reported only when running on CUDA; CPU-only runs show `n/a`

| Stage | Model load (s) | Mean image runtime (s) | Mean image predict (s) | Global patches/s | Global MPix/s | Peak GPU alloc (MiB) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADVENT (DeepLabV3+) | 0.558 | 0.602 | 0.433 | 203.328 | 28.172 | 521.476 | 1356.840 |
