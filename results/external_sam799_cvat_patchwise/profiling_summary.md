# Profiling Summary

Patchwise runtime and memory summary on SAM799-CVAT.

- `runtime_total_sec`: image load + patchwise inference/stitching + threshold/postprocess/metrics; excludes optional asset writes
- `predict_total_sec`: patch preprocessing + batched forward passes + resize-back + probability averaging
- `global_patches_per_sec` and `global_megapixels_per_sec` are computed from aggregate predict time
- GPU peaks are reported only when running on CUDA; CPU-only runs show `n/a`

| Stage | Model load (s) | Mean image runtime (s) | Mean image predict (s) | Global patches/s | Global MPix/s | Peak GPU alloc (MiB) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source-only | 2.839 | 1.302 | 1.127 | 78.055 | 10.815 | 1871.848 | 1446.516 |
| B1 selected | 1.015 | 1.189 | 1.018 | 86.412 | 11.973 | 1872.250 | 1446.977 |
| B2 fs05 | 0.898 | 1.118 | 0.951 | 92.578 | 12.827 | 1872.250 | 1446.883 |
| B2 fs10 | 0.886 | 1.093 | 0.928 | 94.878 | 13.146 | 1872.250 | 1446.238 |
| B2 fs20 | 0.757 | 1.084 | 0.917 | 95.950 | 13.294 | 1872.250 | 1446.426 |
