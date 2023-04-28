PATH_TO_BRAT = "/home/linuxmint/Documents/Brat/brat-1.3p1"
COLLECTION_NAME = "tutorials/bio"
FILE_NAME = "040-text_span_annotation.ann"

class Annotation:
    id = None
    type = None
    begin = None
    end = None
    excerpt = None

    def __init__(self, id, type, begin, end, excerpt) -> None:
        self.id = id
        self.type = type
        self.begin = begin
        self.end = end
        self.excerpt = excerpt

if __name__ == '__main__':
    complete_path = PATH_TO_BRAT + "/data/" + COLLECTION_NAME + "/" + FILE_NAME

    with open(complete_path, 'r') as file:
        lines = file.readlines()
        array = [Annotation(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in lines]
        print(array[0].id)

