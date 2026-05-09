# Main Results

| Model | Stage | Setting | UAV labels | IoU | F1 | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | Source-only | raw | 0 | 0.1284 | 0.2244 | 0.1330 | 0.8040 |
| DeepLabV3+ | Source-only | raw | 0 | 0.1230 | 0.2152 | 0.1303 | 0.6913 |
| SegFormer-B2 | Source-only | raw | 0 | 0.1442 | 0.2476 | 0.1495 | 0.7727 |
| DeepLabV3+ | B1 | raw @ 0.5 | 0 | 0.2830 | 0.4143 | 0.5820 | 0.3767 |
| SegFormer-B2 | B1 | raw @ 0.7 | 0 | 0.3325 | 0.4727 | 0.5374 | 0.5133 |
| DeepLabV3+ | B2 | fs05_pat12 | 9 | 0.3599 | 0.5226 | 0.4967 | 0.5998 |
| SegFormer-B2 | B2 | fs05 | 9 | 0.5074 | 0.6695 | 0.6232 | 0.7277 |
| DeepLabV3+ | B2 | fs10 | 19 | 0.4354 | 0.6005 | 0.5608 | 0.6620 |
| SegFormer-B2 | B2 | fs10 | 19 | 0.5420 | 0.6988 | 0.6762 | 0.7251 |
| DeepLabV3+ | B2 | fs20 | 38 | 0.4760 | 0.6340 | 0.5903 | 0.7060 |
| SegFormer-B2 | B2 | fs20 | 38 | 0.5686 | 0.7209 | 0.6724 | 0.7826 |
| SegFormer-B2 | Upper bound | full-train | 189 | 0.5879 | 0.7369 | 0.6970 | 0.7830 |
