import logging
import os
import platform
from typing import Iterable, Union
from modAL.uncertainty import uncertainty_sampling
import sklearn
import numpy as np
from Annotation import Annotation
from AnnotationFile import AnnotationFile
from TextFile import TextFile

from constants import FOLDER_NAME, PATH_TO_BRAT, COLLECTION_NAME, ANNOTATION_TYPE


class ActiveLearning:
    """
    The ActiveLearning object provides functionalities of Active Learning.
    """

    def __init__(self) -> None:
        pass

    def iteration(self, classifier: sklearn.base.BaseEstimator,
                  unlabeled_data: Union[list, np.ndarray],
                  num_to_annotate: int = 1):
        indices = uncertainty_sampling(classifier=classifier, 
                                       X=unlabeled_data, 
                                       n_instances=num_to_annotate)
        
        if isinstance(unlabeled_data, list):
            unlabeled_data = np.array(unlabeled_data)
        uncertain_samples = list(unlabeled_data[indices])

        return uncertain_samples
    
    def add_samples_to_annotation_files(self, samples: Iterable[str]):
        """
        Add samples to annotation files.

        Arguments
        ---------
            samples (Iterable[str]): Samples to add to annotation files
        """
        operating_system = platform.system()
        slash = "\\" if operating_system == "Windows" else "/"
        collection_path = PATH_TO_BRAT + slash + FOLDER_NAME + slash
        collection_path += COLLECTION_NAME + slash if COLLECTION_NAME else ""

        text_files = [
            TextFile(file_name=file_name, path=collection_path)
            for file_name in os.listdir(collection_path)
            if ".txt" in file_name
        ]
        logging.info(f"Text files found: {str([file.file_name for file in text_files])}")
        
        for file in text_files:
            containing_words = file.contains(excerpts=samples)
            annotations = []
            annotation_file_name = file.file_name[:file.file_name.find(".")] + ".ann"
            for word_info in containing_words:
                annotations.append(Annotation(file_name=annotation_file_name, 
                                              type=ANNOTATION_TYPE, 
                                              begin=word_info[1], 
                                              end=word_info[2], 
                                              excerpt=word_info[0]))
            annotation_file = AnnotationFile(file_name=annotation_file_name, 
                                             path=collection_path)
            annotation_file.add_annotations(annotations=annotations)
        