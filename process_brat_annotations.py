import os
import platform
import numpy as np

import pandas as pd
from ActiveLearning import ActiveLearning
from Dataset import Dataset
from SciBertWordClassifier import SciBertWordClassifier
from modAL import uncertainty
from TextFile import TextFile
from Annotation import Annotation
from AnnotationFile import AnnotationFile
from constants import COLLECTION_NAME, FILE_NAME, FOLDER_NAME, PATH_TO_BRAT

def create_custom_dataset():
    """
    Creates a custom dataset with 50 drug names and 50 no-drug names.

    Returns:
        X (list): List of input word samples.
        y (list): List of target labels (1 for drug, 0 for no-drug).
    """
    # drug_names = [
    #     "aspirin",
    #     "ibuprofen",
    #     "paracetamol",
    #     "simvastatin",
    #     "lisinopril",
    #     "metformin",
    #     "metoprolol",
    #     "levothyroxine",
    #     "amlodipine",
    #     "atorvastatin",
    #     "losartan",
    #     "omeprazole",
    #     "hydrochlorothiazide",
    #     "rosuvastatin",
    #     "warfarin",
    #     "fluoxetine",
    #     "pantoprazole",
    #     "sertraline",
    #     "gabapentin",
    #     "citalopram",
    #     "escitalopram",
    #     "alprazolam",
    #     "venlafaxine",
    #     "amitriptyline",
    #     "lorazepam",
    #     "duloxetine",
    #     "tramadol",
    #     "clonazepam",
    #     "oxycodone",
    #     "phenytoin",
    #     "carbamazepine",
    #     "quetiapine",
    #     "risperidone",
    #     "trazodone",
    #     "olanzapine",
    #     "fluconazole",
    #     "amoxicillin",
    #     "azithromycin",
    #     "doxycycline",
    #     "clindamycin",
    #     "cephalexin",
    #     "sulfamethoxazole",
    #     "amoxicillin-clavulanate",
    #     "levofloxacin",
    #     "ciprofloxacin",
    #     "metronidazole",
    #     "erythromycin",
    #     "valacyclovir",
    #     "acyclovir",
    #     "methotrexate",
    #     "adalimumab"
    # ]


    # no_drug_names = [
    #     "apple",
    #     "car",
    #     "sun",
    #     "book",
    #     "tree",
    #     "table",
    #     "chair",
    #     "computer",
    #     "pen",
    #     "phone",
    #     "house",
    #     "flower",
    #     "dog",
    #     "cat",
    #     "bird",
    #     "fish",
    #     "mountain",
    #     "river",
    #     "ocean",
    #     "sky",
    #     "moon",
    #     "star",
    #     "cloud",
    #     "grass",
    #     "sand",
    #     "rock",
    #     "paper",
    #     "scissors",
    #     "glass",
    #     "wood",
    #     "metal",
    #     "plastic",
    #     "air",
    #     "water",
    #     "fire",
    #     "earth",
    #     "light",
    #     "sound",
    #     "time",
    #     "energy",
    #     "power",
    #     "space",
    #     "color",
    #     "music",
    #     "friend",
    #     "love",
    #     "life",
    #     "smile",
    #     "dream",
    #     "hope"
    # ]


    # Combine drug and no-drug names
    X = drug_names + no_drug_names
    y = [1] * len(drug_names) + [0] * len(no_drug_names)

    return X, y

if __name__ == "__main__":
    operating_system = platform.system()
    slash = "\\" if operating_system == "Windows" else "/"
    collection_path = PATH_TO_BRAT + slash + FOLDER_NAME + slash
    collection_path += COLLECTION_NAME + slash if COLLECTION_NAME else ""

    """
    Reading all annotation files within a directory
    -----------------------------------------------

    files = [
        AnnotationFile(file_name=file_name, path=collection_path)
        for file_name in os.listdir(collection_path)
        if ".ann" in file_name
    ]
    annotation_lists = [file.read() for file in files]
    annotations = [ann for ann_list in annotation_lists for ann in ann_list]

    print([ann.to_string(usage="info") for ann in annotations])
    """

    """
    Writing into a new annotation file
    ----------------------------------

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
    """
    # Creating a cleared dataset for supervised learning out of the annotations
    # -------------------------------------------------------------------------

    dataset = Dataset(path_to_collection=collection_path)
    dataset.to_json(collection_path, "test.jsonl")
    """

    # Example usage of SciBertWordClassifier and ActiveLearning
    # ---------------------------------------------------------

    # num_classes = 2  # Drug and non-drug classes
    # classifier = SciBertWordClassifier(num_classes)

    # X_train, y_train = create_custom_dataset()

    # classifier.fit(X_train, y_train)

    # text = "The patient was prescribed aspirin for pain relief."

    # words = np.array(text.split())

    # predicted_labels = classifier.predict(words)

    # probabilities = classifier.predict_proba(words)

    # for word, label, probability in zip(words, predicted_labels, probabilities):
    #     if label == 1:
    #         print(f"Word: {word}, Class: Drug, Probability: {probability[1]}")
    #     else:
    #         print(f"Word: {word}, Class: Non-Drug, Probability: {probability[0]}")

    # learner = ActiveLearning()
    # uncertainty = uncertainty.classifier_uncertainty(classifier, words)
    # print(uncertainty)

    # uncertain_words = learner.iteration(classifier, words, 3)
    # print(uncertain_words)

    # ann_file = AnnotationFile("example.ann", "C:\\Users\\albbl\\Documents\\Studium\\10_Semester\\Projekt\\")
    # anns = ann_file.read()
    # ann = Annotation("example.ann.txt", "drug", 5, 6, "test")
    # ann_file.add_annotations([ann])

    active_learn = ActiveLearning()
    active_learn.add_samples_to_annotation_files(["case", "substring"])