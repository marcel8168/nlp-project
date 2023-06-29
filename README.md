# Project: Toolset for active learning based control of medical free-text annotations.

This is a project in the context of my master's program at the Friedrich-Alexander-University Nuremberg-Erlangen


## Setup

```
pip install .
python start.py --label {target_label} --label_list {labels}
```

## Command Line Options

The following command line options are available for this application:

- `--label LABEL`: Label of the target class (required).
- `--label_list LABEL_LIST`: List of all possible labels (required).
- `--path PATH`: Path to brat directory (default: PATH_TO_BRAT).
- `--collection COLLECTION`: Name of the annotation collection (default: COLLECTION_NAME).
- `--folder FOLDER`: Name of the destination folder (default: FOLDER_NAME).
- `--num_suggestions NUM`: Number of suggestions to be made by the Active Learning functionality (default: 3).
- `--token_aggregation TOKEN_AGGREGATION`: Strategy used for token probability aggregation (default: max).
- `--iterations NUM`: Number of active learning iterations (default: 10).


## Contribution

The bundled BRAT setup was forked from https://github.com/dtoddenroth/annotationimage. Thanks to @dtoddenroth.
