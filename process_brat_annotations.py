import logging
import os
import platform
import numpy as np

import pandas as pd
from ActiveLearning import ActiveLearning
from Dataset import Dataset
from SciBertWordClassifier import SciBertWordClassifier
from SciBertClassifier import SciBertClassifier
from modAL import uncertainty
from TextFile import TextFile
from Annotation import Annotation
from AnnotationFile import AnnotationFile
from constants import COLLECTION_NAME, FILE_NAME, FOLDER_NAME, PATH_TO_BRAT
from data.drug_dataset import DRUG_NAMES, NO_DRUG_NAMES
from datasets import load_dataset


def create_custom_dataset():
    """
    Creates a custom dataset with 50 drug names and 50 no-drug names.

    Returns:
        X (list): List of input word samples.
        y (list): List of target labels (1 for drug, 0 for no-drug).
    """
    drug_names = DRUG_NAMES
    no_drug_names = NO_DRUG_NAMES

    # Combine drug and no-drug names
    X = drug_names + no_drug_names
    y = [1] * len(drug_names) + [0] * len(no_drug_names)

    return X, y


if __name__ == "__main__":
    operating_system = platform.system()
    slash = "\\" if operating_system == "Windows" else "/"
    collection_path = PATH_TO_BRAT + slash + FOLDER_NAME + slash
    collection_path += COLLECTION_NAME + slash if COLLECTION_NAME else ""

    # Reading all annotation files within a directory
    # -----------------------------------------------
    """
    files = [
        AnnotationFile(file_name=file_name, path=collection_path)
        for file_name in os.listdir(collection_path)
        if ".ann" in file_name
    ]
    annotation_lists = [file.read() for file in files]
    annotations = [ann for ann_list in annotation_lists for ann in ann_list]

    print([ann.to_string(usage="info") for ann in annotations])
    """

    # Writing into a new annotation file
    # ----------------------------------
    """
    test_file = AnnotationFile(file_name=FILE_NAME + ".ann", path=collection_path)
    test_file.write(
        [
            Annotation(
                file_name=FILE_NAME + ".ann",
                id="T1",
                type="Candidate",
                begin=13,
                end=24,
                excerpt="Aspirin",
            ),
            Annotation(
                file_name=FILE_NAME + ".ann",
                id="T2",
                type="Candidate",
                begin=51,
                end=64,
                excerpt="Cetirizin",
            ),
        ]
    )
    annotations = test_file.read()
    print([ann.to_string(usage="info") for ann in annotations])
    """

    # Creating a cleared dataset for supervised learning out of the annotations
    # -------------------------------------------------------------------------
    """
    dataset = Dataset(path_to_collection=collection_path)
    dataset.to_json(collection_path, "test.jsonl")
    """

    # Example usage of SciBertWordClassifier and ActiveLearning
    # ---------------------------------------------------------
    """
    num_classes = 2  # Drug and non-drug classes
    classifier = SciBertWordClassifier(num_classes)

    X_train, y_train = create_custom_dataset()

    classifier.fit(X_train, y_train)

    text = "The patient was prescribed ibuprofen for pain relief."

    words = np.array(text.split())

    predicted_labels = classifier.predict(words)

    probabilities = classifier.predict_proba(words)

    for word, label, probability in zip(words, predicted_labels, probabilities):
        if label == 0:
            print(f"Word: {word}, Class: Drug, Probability: {probability[1]}")
        else:
            print(f"Word: {word}, Class: Non-Drug, Probability: {probability[0]}")

    learner = ActiveLearning()
    uncertainty = uncertainty.classifier_uncertainty(classifier, words)
    print(uncertainty)

    uncertain_words = learner.iteration(classifier, words, 3)
    print(uncertain_words)
    """

    # Test add_samples_to_annotation_files()
    # --------------------------------------
    """
    active_learn = ActiveLearning()
    active_learn.add_samples_to_annotation_files(["Genetic"])
    """

    # Test SciBertClassifier
    # ----------------------
    logging.basicConfig(filename='example.log', filemode='w', encoding='utf-8', level=logging.INFO)
    cons_dataset = load_dataset("json", data_files="./notebooks/data/dataset.jsonl")
    cons_dataset = cons_dataset["train"].train_test_split()

    classifier = SciBertClassifier(3, "drug", ['O', 'B-drug', 'I-drug'])
    labeled_dataset = cons_dataset.map(classifier.generate_row_labels)
    classifier.fit(labeled_dataset)

    learner = ActiveLearning()
    text_file = TextFile("36476732.txt", "doc")
    test_text = text_file.read()
    uncertain_words = learner.iteration(classifier, [test_text], 3)
    