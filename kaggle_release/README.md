# Kaggle UAV Dataset Mirror

This directory mirrors the raw `images/` and `masks/` files from the `UAV-Based Crack Detection Dataset` used as the primary UAV target benchmark in this repository.

The Kaggle UAV crack dataset is redistributed here because the Kaggle dataset page lists its license as `CC0: Public Domain` at the time of access. A license evidence file, access-time capture, source URL, file list, and `SHA-256` checksums are provided for provenance. Users should also consult the original Kaggle page when available.

Contents:

- `images/`: mirrored raw image files from the Kaggle dataset
- `masks/`: mirrored raw binary crack masks from the Kaggle dataset
- `LICENSE_KAGGLE_CC0.txt`: access-time license record and caveats
- `kaggle_license_screenshot.png`: manual screenshot of the Kaggle dataset page showing the observed license label at access time
- `source_url.txt`: original Kaggle page URL and access date
- `filelist.csv`: relative file list for the mirrored data files
- `sha256_manifest.csv`: relative paths, file sizes, and `SHA-256` checksums for the mirrored data files

This mirror intentionally excludes repository-generated split files and few-shot manifests. Use [split_manifests](../split_manifests) for the frozen `train / val / test` protocol files used in the paper.

Redistribution caveat:

- `CC0` does not clear third-party privacy, publicity, trademark, or similar non-copyright rights.
- Review the imagery for people, license plates, addresses, and other identifying content before further redistribution or republication.
