# B1 Hold-Out Summary

Main takeaways:

- `TS-bank thr080_mean082 @ 0.6` is now the strongest `SegFormer-B2 B1` result on the fixed hold-out split with `IoU 0.3775` and `F1 0.5317`.
- Relative to source-only raw, the promoted `TS-bank @ 0.6` row improves IoU by `0.2333` and F1 by `0.2841`.
- Relative to the old main-report `B1 raw @ 0.7` row, the promoted `TS-bank @ 0.6` row adds another `0.0450` IoU and `0.0590` F1.
- Within the promoted TS-bank checkpoint itself, threshold calibration from `0.5 -> 0.6` adds a smaller but still real gain on top of the default raw row (`IoU 0.3728 -> 0.3775`, `precision 0.4935 -> 0.5257`).
- The initial single random-background control on `UAV val` was much lower than the mined TS-bank row (`0.2747 < 0.3693`), but paired `5-seed` validation reruns changed that reading: matched random background averages `IoU 0.3401 +- 0.0215`, while the mined TS-bank averages `0.3194 +- 0.0239`. The promoted hold-out row remains valid, but its mechanism explanation should stay cautious rather than claiming a clear mined-signal advantage.
- Audit-derived curated reruns also stayed below the raw mined validation winner: the strongest curated export (`thr080_mean082 hard_fp`) reaches `IoU 0.3535`, while the matching `hard_fp + ambiguous` export falls to `0.3304`. This makes the `SegFormer-B2` audit useful for diagnosing bank composition, but not for replacing the promoted `B1` row.
- `B1 raw @ 0.9` still reaches the high-precision regime with `precision 0.6497`, but it is now best treated as a comparison operating point rather than as the main reporting row.
- The old deployment switch remains useful for the weaker source-only model (`IoU 0.1442 -> 0.1784`), but adds little once the model has been improved by either raw `B1` calibration or the promoted `TS-bank` row.
