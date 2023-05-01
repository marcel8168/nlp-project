from Annotation import Annotation
from AnnotationFile import AnnotationFile
from constants import COLLECTION_NAME, PATH_TO_BRAT

if __name__ == "__main__":
    collection_path = PATH_TO_BRAT + "/data/" + COLLECTION_NAME + "/"
    """
    files = [
        AnnotationFile(file_name=file_name, path=collection_path)
        for file_name in os.listdir(collection_path)
        if ".ann" in file_name
    ]
    annotation_lists = [file.read() for file in files]
    annotations = [ann for ann_list in annotation_lists for ann in ann_list]

    print([ann.to_string(usage="info") for ann in annotations])
    """

    test_file = AnnotationFile(file_name="test.ann", path=collection_path)
    test_file.write(
        [
            Annotation(
                file_name="test.ann",
                id="T1",
                type="Candidate",
                begin=13,
                end=24,
                excerpt="Aspirin",
            ),
            Annotation(
                file_name="test.ann",
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
