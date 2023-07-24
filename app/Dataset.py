import os
from typing import Optional

import pandas as pd
from AnnotationFile import AnnotationFile
from System import System
from TextFile import TextFile


class Dataset:
    """
    The Dataset object contains data that can be used for training a NLP model.

    Attributes
    ----------
        dataset (DataFrame): Stores all annotated data
    """

    def __init__(self, path_to_collection: Optional[str] = None,
                 annotation_files = None,
                 text_files = None) -> None:
        if path_to_collection:
            self.dataset = self._create_from_collection(path_to_collection=path_to_collection)
        elif annotation_files and text_files:
            self.dataset = self._create_from_files(annotation_files=annotation_files, text_files=text_files)
        else: 
            self.dataset = pd.DataFrame([])

    def _create_from_files(self, annotation_files: list, text_files: list):
        """
        Create a dataset object from files.

        Arguments
        ---------
            annotation_files (list[AnnotationFile]): List of AnnotationFile objects.
            text_files (list[TextFile]): List of TextFile objects.

        Returns
        -------
            DataFrame: Dataset object.

        """
        sys = System()
        target_class = sys.get_constant(constant_name="TARGET_CLASS")

        dataset = []
        for text_file in text_files:
            annotation_file = list(filter(lambda file: file.file_name[:-4] == text_file.file_name[:-4], annotation_files))[0]
            annotations = annotation_file.read(filter=target_class)
            sentence_info = text_file.get_sentence_info()
            if annotations and not sentence_info.empty:
                for idx in sentence_info.index:
                    included_annotations = [annotation for annotation in annotations if annotation.begin >= sentence_info["start"][idx] and annotation.end <= sentence_info["end"][idx]]
                    if not included_annotations:
                        continue
                    included_excerpts = [annotation.excerpt for annotation in included_annotations]
                    included_start_indices = [annotation.begin - sentence_info["start"][idx] for annotation in included_annotations]
                    included_end_indices = [annotation.end - sentence_info["start"][idx] for annotation in included_annotations]
                    data = {"text":sentence_info["sentence"][idx],
                            "drug":included_excerpts,
                            "drug_indices_start": included_start_indices,
                            "drug_indices_end": included_end_indices
                            }
                    dataset.append(data)
        dataset = pd.DataFrame(data=dataset)
        self.dataset = dataset
        return dataset
        
    def _create_from_collection(self, path_to_collection: str) -> pd.DataFrame:
        """
        Create a dataset object from the collection path.

        Arguments
        ---------
            path_to_collection (str): Path to the collection directory.

        Returns
        -------
            DataFrame: Dataset object.

        """
        files = os.listdir(path=path_to_collection)
        annotation_files = [
            AnnotationFile(file_name=file, path=path_to_collection)
            for file in files
            if ".ann" in file
        ]
        
        text_files = [
            TextFile(file_name=file, path=path_to_collection)
            for file in files
            if ".txt" in file
        ]

        return self._create_from_files(annotation_files=annotation_files, text_files=text_files)

    def to_json(self, path, name) -> None:
        """
        Convert the dataset object to a JSON-File.

        Arguments
        ---------
            path (str): Path to the storage location.
            name (str): File name.

        """
        self.dataset.to_json(path_or_buf=path + name, orient="records", lines=True)