import logging
from ActiveLearning import ActiveLearning
from SciBertClassifier import SciBertClassifier

from System import System
from TextFile import TextFile
from constants import COLLECTION_NAME, FOLDER_NAME, PATH_TO_BRAT


if __name__ == "__main__":
        logging.basicConfig(filename='example.log', filemode='w', encoding='utf-8', level=logging.INFO)

        system = System()
        path_to_collection, file_names = system.get_file_names_from_path(path_to_brat=PATH_TO_BRAT, folder_name=FOLDER_NAME, collection_name=COLLECTION_NAME)
        texts = []
        for file_name in file_names:
            if ".txt" in file_name:
                texts.append(TextFile(file_name=file_name, path=path_to_collection).read())

        classifier = SciBertClassifier(num_classes=3, label="drug", label_list=['O', 'B-drug', 'I-drug'])

        active_learner = ActiveLearning()
        active_learner.iteration(classifier=classifier, unlabeled_data=texts[:1], num_to_annotate=3)