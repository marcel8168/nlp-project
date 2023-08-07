# Project: Toolset for active learning based control of medical free-text annotations [DOCKERIZED VERSION]

This is a project in the context of my master's program at the Friedrich-Alexander-University Nuremberg-Erlangen
Please note that this project has only been tested on Windows and Linux operating systems. Nevertheless, macOS was also considered in the implementations.

## Setup

1. Clone this repository
```
    git clone -b dockerized-tool-set https://github.com/marcel8168/nlp-project.git
```
2. Start the Docker Engine
3. Build and run docker container
```
    docker compose up
```
4. Open the browser
```
    http://localhost:8001/
```
5. Under 'network options' activate the autorefresh option.


## Instructions for use

The text files to be annotated must be placed in the doc/ folder. As an example, there are two example text files and a corresponding annotation file. Please remove them if you want to apply the program to your own texts. Otherwise, the example files will be included in the training process.\
After running the start.py script, BRAT is started on a local server and the frontend is automatically opened in the browser. Meanwhile, the model is applied on the text files and computes suggestions to be annotated according to the uncertainty sampling strategy. After all suggestions are loaded, the user is notified and can start annotating. The suggestions will be displayed in red with the entity type TO_BE_ANNOTATED. Any change will be applied to all identical cases and therefore BRAT will be reloaded to display all assignments. The user can also make annotations other than suggested. This is typically useful at the beginning to increase the data set for the first training loops.\
After all suggestions have been changed, the user will be notified about the next training iteration and the current prediction performance will be displayed. It is recommended not to make any changes during the training, as they will be taken into account only in the next training step. Please note that until now all pop-ups must be confirmed or closed to continue the process.


## Developer Options

#### Command Line Options

The following command line options are available for this application. Note that these must be changed in the dockerfile:

- `--label LABEL`: Label of the target class (required).
- `--label_list LABEL_LIST`: List of all possible labels (required).
- `--path PATH`: Path to brat directory (default: PATH_TO_BRAT).
- `--collection COLLECTION`: Name of the annotation collection (default: COLLECTION_NAME).
- `--folder FOLDER`: Name of the destination folder (default: FOLDER_NAME).
- `--num_suggestions NUM`: Number of suggestions to be made by the Active Learning functionality (default: 10).
- `--token_aggregation TOKEN_AGGREGATION`: Strategy used for token probability aggregation (default: max).
- `--iterations NUM`: Number of active learning iterations (default: 10).
- `--certainty_threshold NUM`: Probability threshold for considering a prediction as certain and adding the annotation to the annotation files (default: 0.9).


#### Active Learning strategy
The currently implemented active learning strategy is based on modAL. Other sampling strategies can be used by replacing the function call in ActiveLearning.py specifically in line 34 of the iteration() function:
```
indices = uncertainty_sampling(classifier=classifier, 
                               X=unlabeled_data, 
                               n_instances=num_to_annotate)
```

#### Whole word prediction
The used tokenizer can split words. For this reason a function whole_word_prediction() was implemented. This function supports the aggregation strategies max (takes the maximum probabilities of all tokens) and avg (takes the average token probability). To change the default strategy max, run the script with the argument:
```
python start.py [...] --token_aggregation avg
```


## Contribution

The bundled BRAT setup was forked from https://github.com/dtoddenroth/annotationimage. Thanks to @dtoddenroth.
