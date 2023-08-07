from itertools import chain
import logging
import os
import re
import time
from typing import Iterable, Union
from modAL.uncertainty import uncertainty_sampling
import sklearn
import numpy as np
from datasets import load_dataset
from Annotation import Annotation
from AnnotationFile import AnnotationFile
from Dataset import Dataset
from System import System
from TextFile import TextFile

from constants import CERTAINTY_THRESHOLD, DATA_PATH, EXTERNAL_TEST_DATASET_FILE_NAME, FOLDER_NAME, PATH_TO_BRAT, COLLECTION_NAME, SUGGESTION_ANNOTATION_TYPE, TRAINING_DATASET_FILE_NAME


class ActiveLearning:
    """
    The ActiveLearning object provides functionalities of Active Learning.
    """

    def __init__(self):
        pass

    def iteration(self, classifier: sklearn.base.BaseEstimator,
                  unlabeled_data: Union[list, np.ndarray],
                  num_to_annotate: int = 1):
        system = System()
        
        sample_lists = []
        for data in unlabeled_data:
            sample_lists.append(uncertainty_sampling(classifier=classifier, 
                                       X=[data], 
                                       n_instances=int(np.ceil(num_to_annotate * 5 / len(unlabeled_data)))))

        samples = (np.array(list(chain(*[sublist[0] for sublist in sample_lists]))).astype(int), list(chain(*[sublist[1] for sublist in sample_lists])))
        samples_indices_sorted = np.argsort(samples[1])[::-1]
        indices_all = samples[0][samples_indices_sorted]
        while num_to_annotate > 0 and indices_all.size > 0:
            indices = indices_all[:num_to_annotate]
            indices_all = indices_all[num_to_annotate:]
            predictions = classifier.predictions.flatten()
            uncertain_samples = list(filter(lambda x: x['index'] in indices, predictions))
            logging.info(f"Suggested samples to be annotated: {uncertain_samples}")
            suggested_samples = list({sample["word"] for sample in uncertain_samples})
            num_to_annotate -= self.add_samples_to_annotation_files(samples=suggested_samples)

        most_certain_predictions = self.get_most_certain_predictions(classifier=classifier, X=unlabeled_data)
        self.add_samples_to_annotation_files(samples=most_certain_predictions)

        path_to_collection, file_names = system.get_file_names_from_path(path_to_brat=PATH_TO_BRAT, folder_name=FOLDER_NAME, collection_name=COLLECTION_NAME)
        file_names = [file_name for file_name in file_names if ".ann" in file_name or ".txt" in file_name]
        while self.suggestions_left_in_files(path=path_to_collection, file_names=file_names):
            annotation_files = [file_name for file_name in file_names if ".ann" in file_name]
            changed_file = self.check_file_change(path=path_to_collection, file_names=annotation_files)
            self.apply_annotation(path=path_to_collection, file_names=file_names, changed_file=changed_file)

        dataset = Dataset(path_to_collection=path_to_collection)
        if not dataset.dataset.empty:
            dataset.to_json(DATA_PATH, TRAINING_DATASET_FILE_NAME)
            logging.info(f"Updated dataset with new annotations is generated and saved under {DATA_PATH + TRAINING_DATASET_FILE_NAME}")
            dataset = load_dataset("json", data_files=DATA_PATH + TRAINING_DATASET_FILE_NAME)
            split_dataset = dataset["train"].train_test_split()
            labeled_dataset = split_dataset.map(classifier.generate_row_labels)

            logging.info("Training with updated and labeled dataset started..")
            classifier.fit(labeled_dataset)
            logging.info("Training finished!")

            classifier.save()
            logging.info("Pretrained model saved.")

        classifier.performance_report(path_to_test_set=DATA_PATH + EXTERNAL_TEST_DATASET_FILE_NAME)
    
    def add_samples_to_annotation_files(self, samples: Iterable[str]) -> int:
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

        samples_added = set()

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
            added = annotation_file.add_annotations(annotations=annotations)
            samples_added = samples_added.union(added)

        return len(samples_added)

    def check_file_change(self, path: str, file_names: list) -> str:
        """
        Checks if a file has been changed by comparing its modification time.

        Arguments
        ---------
            filename (str): The path to the file to monitor.
        """
        initial_state = {}

        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            with open(file_path, 'r') as file:
                initial_state[file_name] = file.read()

        while True:
            time.sleep(1)
            for file_name in file_names:
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r') as file:
                    if file.read() != initial_state[file_name]:
                        return file_name
    
    def suggestions_left_in_files(self, path: str, file_names: list) -> bool:
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
    
    def apply_annotation(self, path: str, file_names: str, changed_file: str) -> None:
        """
        Applies annotations from an old version of the text file to all identical words
        in the new version of the text file, including the label, begin index, and end index.

        Arguments
        ---------
            path (str): Path to the collection.
            file_names (str): Files in the collection.
            changed_file (str): Annotation file that was changed.

        Raises
        ------
            ValueError: If changed file is not of type '.ann'
        """
        if ".ann" not in changed_file:
            raise ValueError("The changed file must be of type '.ann'")
        
        distinct_file_names = {file_name[:-4] for file_name in file_names if ".txt" in file_name}
        annotation_file = AnnotationFile(file_name=changed_file, path=path)
        annotations = [ann for ann in annotation_file.read() if ann.type != SUGGESTION_ANNOTATION_TYPE]
        for annotation in annotations:
            excerpt = annotation.excerpt

            for text_file in distinct_file_names:
                sentences = TextFile(file_name=text_file + ".txt", path=path).get_sentence_info()
                new_annotations= []

                for idx, sentence in enumerate(sentences["sentence"]):
                    pattern = r'\b{}\b'.format(excerpt)
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        new_annotations.append(
                            Annotation(file_name=text_file + ".ann",
                                        type=annotation.type,
                                        begin=sentences["start"][idx] + match.start(),
                                        end=sentences["start"][idx] + match.end(),
                                        excerpt=match.group(0)
                                    )
                            )
                annotation_file_to_change = AnnotationFile(file_name=text_file + ".ann", path=path)
                annotation_file_to_change.add_annotations(annotations=new_annotations, overwrite_existing=True)

    def get_most_certain_predictions(self, classifier: sklearn.base.BaseEstimator, X: Iterable):
        """
        Get the most certain predictions from the model's predictions.

        Arguments
        ---------
            classifier (sklearn.base.BaseEstimator): Classifier to compute predictions.
            X (Iterable): Input data to make predictions.

        Returns
        -------
            numpy.ndarray: Array containing the most certain predictions.

        """
        probabilities = classifier.predict(X=X).flatten()
        probabilities = list(chain(*probabilities))
        most_certain_predictions = {x['word'] for x in probabilities if x['score'] > CERTAINTY_THRESHOLD and x['entity'] != 'LABEL_0'}

        return most_certain_predictions