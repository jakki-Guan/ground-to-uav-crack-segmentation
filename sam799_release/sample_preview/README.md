# Sample Preview

This directory contains two lightweight, downsampled preview cases for the metadata-only `SAM799-CVAT` release stub.

Files included per case:

- `*_image.png`: downsampled RGB preview
- `*_mask.png`: binary crack annotation mask
- `*_overlay.png`: RGB preview with the crack annotation overlaid

Selected cases:

1. `preview_01_dji_0042`
   - Source image id: `DJI_0042_JPG.rf.FGMquUU7t2bAR7l6zz7k`
   - Why selected: representative single-crack roadway case with painted-marking context
   - Files:
     - `preview_01_image.png`
     - `preview_01_mask.png`
     - `preview_01_overlay.png`

2. `preview_02_dji_0057`
   - Source image id: `DJI_0057_JPG.rf.7Rq9A6jCjWCEukKBcgtq`
   - Why selected: representative multi-crack case near curb / island geometry with shadow variation
   - Files:
     - `preview_02_image.png`
     - `preview_02_mask.png`
     - `preview_02_overlay.png`

Notes:

- These previews are illustrative examples only and do not constitute the full dataset release.
- The preview images are downsampled and re-encoded as PNG files without the original image metadata.
- If a later privacy or redistribution review flags an issue, replace or remove the affected preview files.
