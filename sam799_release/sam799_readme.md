# SAM799-CVAT Metadata-Only Release Stub

This directory is a metadata-only public-release stub for the self-collected `SAM799-CVAT` UAV crack dataset used as an external-check benchmark in the JSTARS manuscript workflow.

Current release status:

- Raw images: not mirrored in this repository
- Raw masks: not mirrored in this repository
- Public dataset host: pending
- Privacy review: pending
- Property / redistribution review: pending
- Release license: pending

Contents:

- `sam799_metadata.csv`: dataset-level summary for the current fixed external-check set
- `sam799_filelist.csv`: per-image file list and lightweight mask statistics
- `sample_preview/`: small downsampled preview bundle with two illustrative image / mask / overlay cases

Current fixed evaluation set:

- split role: `external_test`
- images: `53`
- protocol role: external validation only
- training / threshold selection on this dataset: `false`

Public-release recommendation:

- Keep the full dataset outside GitHub and publish it through a dataset-oriented host such as `Zenodo`, `OSF`, or `Hugging Face Datasets`.
- Use this repository only for metadata, file lists, sample previews, and the eventual dataset link.

Checklist before publishing the full dataset:

1. Review every image for personally identifiable or sensitive visual content such as faces, license plates, addresses, or private-property markers.
2. Confirm the original UAV collection and redistribution plan is consistent with applicable flight, privacy, campus, municipal, and property-use rules.
3. Decide whether preview images need redaction before public release.
4. Choose and document a release license.
