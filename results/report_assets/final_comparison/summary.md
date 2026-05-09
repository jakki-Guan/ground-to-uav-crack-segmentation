# Final Comparison Summary

Main takeaways:

- The raw-only main table keeps the formal reporting rule intact: deployment/postprocess operating points are excluded from the main comparison.
- Among frozen source-only baselines on the fixed UAV hold-out, `SegFormer-B2` is strongest (`IoU 0.1442`), followed by `U-Net` (`0.1284`) and `DeepLabV3+` (`0.1230`).
- `B1` improves both transferable models at zero target labels: `SegFormer-B2` gains `+0.1883` IoU and `DeepLabV3+` gains `+0.1599`.
- The few-shot curve stays monotonic for both models. By `fs20`, `SegFormer-B2` reaches `IoU 0.5686` and `DeepLabV3+` reaches `IoU 0.4760`.
- `SegFormer-B2 fs20` reaches about `96.7%` of the frozen in-domain upper bound (`IoU 0.5879`), which confirms that the remaining gap is largely correctable with modest UAV annotation.
- The qualitative diagnosis figure focuses on representative SegFormer source-only failures that exhibit high recall, low precision, and structured false positives on UAV backgrounds: `slide1290, slide1434, slide1089`.
