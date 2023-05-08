import os
from typing import Optional

import pandas as pd
from AnnotationFile import AnnotationFile
from TextFile import TextFile


class Dataset:
    """
    The Dataset object contains data that can be used for training a NLP model.

    Attributes
    ----------
        dataset []
    """

    def __init__(self, path_to_collection: Optional[str] = None, 
                 annotation_files: Optional[list[AnnotationFile]] = None, 
                 text_files: Optional[list[TextFile]] = None) -> None:
        if path_to_collection:
            self.dataset = self._create_from_collection(path_to_collection=path_to_collection)
        elif annotation_files and text_files:
            self.dataset = self._create_from_files(annotation_files=annotation_files, text_files=text_files)
        else: 
            self.dataset = pd.DataFrame([])

    def _create_from_files(self, annotation_files: list[AnnotationFile], text_files: list[TextFile]) -> pd.DataFrame:
        """
        """
        annotation_lists = [file.read() for file in annotation_files]
        annotation_lists = filter(None, annotation_lists)
        dataset = []
        for annotation_list in annotation_lists:
            text_file = list(filter(lambda file: file.file_name[:-4] == annotation_list[0].file_name[:-4], text_files))[0]
            sentence_info = text_file.get_sentence_info()
            for annotation in annotation_list:
                text_excerpt_info = sentence_info.query(f"start <= {annotation.begin} <= end")
                data = [text_excerpt_info["sentence"].values[0], annotation.excerpt, annotation.begin - text_excerpt_info["start"].values[0], annotation.end - text_excerpt_info["start"].values[0]]
                dataset.append(data)
        dataset = pd.DataFrame(data=dataset)
        self.dataset = dataset
        return dataset
        
    def _create_from_collection(self, path_to_collection: str) -> pd.DataFrame:
        """
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
        self.dataset.to_json(path_or_buf=path + name, orient="records", lines=True)