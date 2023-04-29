import os


PATH_TO_BRAT = "/home/linuxmint/Documents/Brat/brat-1.3p1"
COLLECTION_NAME = "tutorials/bio"
FILE_NAME = "040-text_span_annotation.ann"

class Annotation:
    '''
    The Annotation object contains information regarding a text annotation.

    Attributes:
        file_name (str): Name of the file where the annotation is stored in,
        id (str): ID of the annotation,
        type (str): Type of the annotation,
        begin (int): Start index of annotated string,
        end (int): End index of annotated string,
        excerpt (str): Excerpt of annotated string
    '''
    def __init__(self, file_name, id, type, begin, end, excerpt) -> None:
        self.file_name = file_name
        self.id = id
        self.type = type
        self.begin = begin
        self.end = end
        self.excerpt = excerpt

    def to_string(self) -> str:
        """
        Create string with all attributes of the Annotation object.

        Returns:
            str: Annotation object as string.
        """
        return f"Annotation({self.file_name}, {self.id}, {self.type}, {self.begin}, {self.end}, {self.excerpt})"


if __name__ == '__main__':
    complete_path = PATH_TO_BRAT + "/data/" + COLLECTION_NAME + "/"

    file_name_list = [file_name for file_name in os.listdir(complete_path) if ".ann" in file_name]
    annotations = set()
    for file_name in file_name_list:
        with open(complete_path + file_name, 'r') as file:
            lines = [line for line in file.readlines() if line[0] == "T"]
            annotations = annotations.union(set(Annotation(file_name, line.split()[0], line.split()[1], line.split()[2],
                                                           line.split()[3], line.split()[4]) for line in lines))
            
    print([ann.to_string() for ann in annotations])

