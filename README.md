# CVAT-Utils

Custom utils for downloading and uploading data in CVAT.

## Installation
Install `cvat-utils` package for production.
```bash
pip install -U setuptools build
python -m build
pip install dist/<package_name>.tar.gz
```

Install `cvat-utils` package for development in editable mode.
```bash
pip install --editable .
```

## Usage
Before using `cvat-utils` create `.env` file with `CVAT_USERNAME` and `CVAT_PASSWORD` for accessing CVAT data.
Example:
```
CVAT_USERNAME=...
CVAT_PASSWORD=...
```

The `cvat-utils` package can be used either as CLI script or a python module.
* Example of CLI script usage in bash: `cvat-utils --help`
* Example of a python module usage in a python script: `import cvat-utils`

### Downloading data from CVAT

Download image metadata and all types of annotations:
```bash
cvat-utils download \
  --task-ids [1,2,3] \
  --output-path output/dataset-1
```

Download image metadata and all types of annotations and images:
```bash
cvat-utils download \
  --task-ids [1,2,3] \
  --output-path output/dataset-1 \
  --load-images
```

Download image metadata and annotations of type `points` and `bboxes`:
```bash
cvat-utils download \
  --task-ids [1,2,3] \
  --output-path output/dataset-1 \
  --points \
  --bboxes
```
Similarly, use any combination of arguments `--points`, `--polylines`, `--polygons`, `--masks`, `--rectangles`, `--bboxes`, and `--tags`.
