import logging
import re
from typing import Iterable, Optional
from os import path
from Annotation import Annotation
from constants import SUGGESTION_ANNOTATION_TYPE


class AnnotationFile:
    """
    The AnnotationFile object contains information of an annotation file.

    Attributes
    ----------
        file_name (str): Name of the file (with file extension) where the annotation is stored in,
        path (str): Location where the file is stored at
    """

    def __init__(self, file_name: str, path: str) -> None:
        self.file_name = file_name
        self.path = path

    def read(self, filter: Optional[str] = None):
        """
        Read the content of the annotation file object.
        
        Arguments
        ----------
            filter (Optional[str]): The type of annotations to filter. If provided, only Annotation objects with a matching type will be included in the returned list. (default: None)

        Returns
        -------
            annotations (set[Annotation]): Set of Annotation objects.
        """
        annotations = []
        full_path = self.path + self.file_name

        if path.exists(full_path):
            with open(full_path, "r", encoding="utf8") as file:
                lines = [line for line in file.readlines() if line.startswith("T")]
                annotations = [
                    Annotation(
                        file_name=self.file_name,
                        id=line.split()[0],
                        type=line.split()[1],
                        begin=int(line.split()[2]),
                        end=int(line.split()[3]),
                        excerpt=line.split()[4],
                    )
                    for line in lines
                    if filter is None or line.split()[1] == filter
                ]

        return annotations

    def write(self, annotations: Iterable[Annotation]) -> None:
        """
        Write into the the annotation file object.

        Arguments
        ---------
            annotations (Iterable[Anntation]): Iterable set of annotation objects
        """
        full_path = self.path + self.file_name

        with open(full_path, "a", encoding="utf8") as file:
            annotation_lines = [
                ann.to_string(usage="annotation") for ann in annotations
            ]
            file.writelines(annotation_lines)
        logging.info(f"Wrote into {self.path + self.file_name}:\n" + str([ann.to_string(usage="info") for ann in annotations]))

    def add_annotations(self, annotations: Iterable[Annotation], overwrite_existing: Optional[bool] = False) -> set[str]:
        """
        Add annotations to the annotation file object.

        Arguments
        ---------
            annotations (Iterable[Anntation]): Iterable set of annotation objects
            overwrite_existing (Optional[bool]): Flag whether to overwrite existing annotations
        """
        existing_annotations = self.read()

        # only add new annotation that do not intersect other annotations
        if existing_annotations:
            new_annotations = {
                annotation
                for annotation in annotations
                if (
                    annotation.id not in {ann.id for ann in existing_annotations} and
                    not any(
                        (annotation.begin >= existing_annotation.begin and annotation.begin <= existing_annotation.end) or
                        (annotation.end >= existing_annotation.begin and annotation.end <= existing_annotation.end) or
                        (annotation.begin <= existing_annotation.begin and annotation.end >= existing_annotation.end)
                        for existing_annotation in existing_annotations
                    )
                )
            }
        else:
            new_annotations = set(annotations)

        # set id
        highest_id_num = max(int(re.search(r"\d+", ann.id).group()) for ann in existing_annotations) if existing_annotations else 0
        for annotation in new_annotations:
            if not annotation.id:
                highest_id_num += 1 
                annotation.id = "T" + str(highest_id_num)
        if new_annotations:
            logging.info("Adding annotations.")
            self.write(new_annotations)

        # overwriting existing annotations
        if overwrite_existing:
            existing_annotations = self.read()
            for existing_ann in existing_annotations[:highest_id_num]:
                for new_ann in annotations:
                    if existing_ann.excerpt == new_ann.excerpt:
                        existing_ann.type = new_ann.type
            annotation_lines = [
                ann.to_string(usage="annotation") for ann in existing_annotations
            ]
            with open(self.path + self.file_name, 'w') as file:
                file.writelines(annotation_lines)
        
        return set(ann.excerpt.lower() for ann in new_annotations)
                
    def suggestions_left(self) -> bool:
        """
        Checks if there are any suggestions left in the annotation file.

        Returns
        -------
            bool: True if there are suggestions left in the file, False otherwise.
        """
        full_path = self.path + self.file_name

        with open(full_path, "r", encoding="utf-8") as file:
            if any(SUGGESTION_ANNOTATION_TYPE in line.strip() for line in file):
                return True

        return False
