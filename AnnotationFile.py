from typing import Iterable
from os import path
from Annotation import Annotation


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

    def read(self) -> list[Annotation]:
        """
        Read the content of the annotation file object.

        Returns
        -------
            set[Annotation]: Set of Annotation objects.
        """
        annotation_lines = []
        full_path = self.path + self.file_name
        if path.exists(full_path):
            with open(full_path, "r", encoding="utf8") as file:
                lines = [line for line in file.readlines() if line.startswith("T")]
                annotation_lines = [
                    Annotation(
                        file_name=self.file_name,
                        id=line.split()[0],
                        type=line.split()[1],
                        begin=int(line.split()[2]),
                        end=int(line.split()[3]),
                        excerpt=line.split()[4],
                    )
                    for line in lines
                ]

        return annotation_lines

    def write(self, annotations: Iterable[Annotation]) -> None:
        """
        Write into the the annotation file object.

        Arguments:
        ---------
            annotations (Iterable[Anntation]): Iterable set of annotation objects
        """
        full_path = self.path + self.file_name

        with open(full_path, "a", encoding="utf8") as file:
            annotation_lines = [
                ann.to_string(usage="annotation") for ann in annotations
            ]
            file.writelines(annotation_lines)

    def add_annotations(self, annotations: Iterable[Annotation]) -> None:
        """
        Add annotations to the annotation file object.

        Arguments:
        ---------
            annotations (Iterable[Anntation]): Iterable set of annotation objects
        """
        existing_annotations = self.read()

        if existing_annotations:
            new_annotations = [
                annotation
                for annotation in annotations
                if (
                    annotation.id not in {ann.id for ann in existing_annotations} and
                    annotation.excerpt not in {ann.excerpt for ann in existing_annotations} and 
                    not any(
                        (annotation.begin >= existing_annotation.begin and annotation.begin <= existing_annotation.end) or
                        (annotation.end >= existing_annotation.begin and annotation.end <= existing_annotation.end) or
                        (annotation.begin <= existing_annotation.begin and annotation.end >= existing_annotation.end)
                        for existing_annotation in existing_annotations
                    )
                )
            ]
        else:
            new_annotations = annotations

        highest_id_num = max(int(ann.id[1]) for ann in existing_annotations) if existing_annotations else 0
        for annotation in annotations:
            if not annotation.id:
                highest_id_num += 1 
                annotation.id = "T" + str(highest_id_num)

        self.write(new_annotations)