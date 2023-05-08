import os
from Dataset import Dataset

from TextFile import TextFile
from Annotation import Annotation
from AnnotationFile import AnnotationFile
from constants import COLLECTION_NAME, FILE_NAME, FOLDER_NAME, PATH_TO_BRAT

if __name__ == "__main__":
    
    collection_path = PATH_TO_BRAT + "/" + FOLDER_NAME + "/"
    collection_path += COLLECTION_NAME + "/" if COLLECTION_NAME else ""

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

    # Creating a cleared dataset for supervised learning out of the annotations
    # -------------------------------------------------------------------------

    dataset = Dataset(path_to_collection=collection_path)
    dataset.to_json(collection_path, "test.json")
    