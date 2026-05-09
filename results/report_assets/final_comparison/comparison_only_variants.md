# Comparison-Only Operating Points

| Model | Stage | Setting | UAV labels | IoU | F1 | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepLabV3+ | Source-only | deploy | 0 | 0.1220 | 0.2090 | 0.1447 | 0.4711 |
| SegFormer-B2 | Source-only | deploy | 0 | 0.1784 | 0.2900 | 0.2099 | 0.5081 |
| SegFormer-B2 | B1 | raw @ 0.5 | 0 | 0.3222 | 0.4677 | 0.4660 | 0.5662 |
| SegFormer-B2 | B1 | raw @ 0.9 | 0 | 0.3193 | 0.4502 | 0.6497 | 0.4112 |
| DeepLabV3+ | B1 | deploy | 0 | 0.1996 | 0.3123 | 0.8021 | 0.2130 |
| SegFormer-B2 | B1 | deploy | 0 | 0.3166 | 0.4469 | 0.6490 | 0.4071 |
