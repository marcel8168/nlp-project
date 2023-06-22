import logging
import os
import time
from typing import Iterable, Union
from modAL.uncertainty import uncertainty_sampling
import sklearn
import numpy as np
from datasets import load_dataset
from Annotation import Annotation
from AnnotationFile import AnnotationFile
from Dataset import Dataset
from Gui import GUI
from System import System
from TextFile import TextFile

from constants import FOLDER_NAME, PATH_TO_BRAT, COLLECTION_NAME, SUGGESTION_ANNOTATION_TYPE


class ActiveLearning:
    """
    The ActiveLearning object provides functionalities of Active Learning.
    """

    def __init__(self):
        pass

    def iteration(self, classifier: sklearn.base.BaseEstimator,
                  unlabeled_data: Union[list, np.ndarray],
                  num_to_annotate: int = 1):
        gui = GUI()
        system = System()

        indices = uncertainty_sampling(classifier=classifier, 
                                       X=unlabeled_data, 
                                       n_instances=num_to_annotate)
        
        predictions = classifier.predictions.flatten()
        uncertain_samples = list(predictions[indices[0]])
        logging.info(f"Suggested samples to be annotated: {uncertain_samples}")
        suggested_samples = [sample["word"] for sample in uncertain_samples]
        self.add_samples_to_annotation_files(samples=suggested_samples)

        title = "Suggestions loaded"
        message = "Suggestions has been loaded.\nYou can now start annotating."
        gui.show_custom_popup(title, message)

        path_to_collection, file_names = system.get_file_names_from_path(path_to_brat=PATH_TO_BRAT, folder_name=FOLDER_NAME, collection_name=COLLECTION_NAME)
        
        while self.suggestions_left_in_files(path=path_to_collection, file_names=file_names):
            self.check_file_change(path=path_to_collection, file_names=file_names)
        logging.info("Annotation by domain expert finished. No suggestions left.")
        title = "Annotation finished"
        message = "You finished the current annotation step.\nNow the next training iteration began.\nPlease do not change any file until the next call."
        gui.show_custom_popup(title, message)

        dataset = Dataset(path_to_collection=path_to_collection)
        storage_path = "./data/" 
        file_name = "training_dataset.json"
        dataset.to_json(storage_path, file_name)
        logging.info(f"Updated dataset with new annotations is generated and saved under {storage_path + file_name}")
        dataset = load_dataset("json", data_files=storage_path + file_name)
        split_dataset = dataset["train"].train_test_split()
        labeled_dataset = split_dataset.map(classifier.generate_row_labels)

        logging.info("Training with updated and labeled dataset started..")
        classifier.fit(labeled_dataset)
        logging.info("Training finished!")

        classifier.save()
        logging.info("Pretrained model saved.")
    
    def add_samples_to_annotation_files(self, samples: Iterable[str]) -> None:
        """
        Adds samples to annotation files.

        Arguments
        ---------
            samples (Iterable[str]): Samples to add to annotation files
        """
        system = System()
        path_to_collection, file_names = system.get_file_names_from_path(path_to_brat=PATH_TO_BRAT, folder_name=FOLDER_NAME, collection_name=COLLECTION_NAME)

        text_files = [
            TextFile(file_name=file_name, path=path_to_collection)
            for file_name in file_names
            if ".txt" in file_name
        ]
        logging.info(f"Text files found: {str([file.file_name for file in text_files])}")
        
        for file in text_files:
            containing_words = file.contains(excerpts=samples)
            annotations = []
            annotation_file_name = file.file_name[:file.file_name.find(".")] + ".ann"
            for word_info in containing_words:
                annotations.append(Annotation(file_name=annotation_file_name, 
                                              type=SUGGESTION_ANNOTATION_TYPE, 
                                              begin=word_info[1], 
                                              end=word_info[2], 
                                              excerpt=word_info[0]))
            annotation_file = AnnotationFile(file_name=annotation_file_name, 
                                             path=path_to_collection)
            annotation_file.add_annotations(annotations=annotations)

    def check_file_change(self, path: str, file_names: list[str]) -> bool:
        """
        Checks if a file has been changed by comparing its modification time.

        Arguments
        ---------
            filename (str): The path to the file to monitor.
        """
        initial_mod_time = np.ndarray((len(file_names),))
        current_mod_time = np.ndarray((len(file_names),))
        
        for idx, file_name in enumerate(file_names):
            initial_mod_time[idx] = os.path.getmtime(path + file_name)
            current_mod_time[idx] = os.path.getmtime(path + file_name)

        while np.array_equal(current_mod_time, initial_mod_time):
            time.sleep(2)
            for idx, file_name in enumerate(file_names):
                current_mod_time[idx] = os.path.getmtime(path + file_name)
        
        return True
    
    def suggestions_left_in_files(self, path: str, file_names: list[str]) -> bool:
        """
        Checks if there are any suggestions left in the given files.

        Arguments
        ---------
            path (str): The path to the directory containing the files.
            file_names (List[str]): A list of file names to check for suggestions.

        Returns
        -------
            bool: True if there are suggestions left in any of the files, False otherwise.
        """
        suggestions = []
        
        for file_name in file_names:
            if ".ann" in file_name:
                annotation_list = AnnotationFile(file_name=file_name, path=path).read()
                suggestions.extend([annotation for annotation in annotation_list if annotation.type == SUGGESTION_ANNOTATION_TYPE])
        
        return len(suggestions) > 0
