from typing import Iterable

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

        with open(self.path + self.file_name, "r", encoding="utf8") as file:
            lines = [line for line in file.readlines() if line[0] == "T"]
            annotation_lines = [
                Annotation(
                    self.file_name,
                    line.split()[0],
                    line.split()[1],
                    int(line.split()[2]),
                    int(line.split()[3]),
                    line.split()[4],
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
        with open(self.path + self.file_name, "w", encoding="utf8") as file:
            annotation_lines = [
                ann.to_string(usage="annotation") for ann in annotations
            ]
            file.writelines(annotation_lines)
