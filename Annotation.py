class Annotation:
    """
    The Annotation object contains information regarding a text annotation.

    Attributes
    ----------
        file_name (str): Name of the file (with file extension) where the annotation is stored in,
        id (str): ID of the annotation,
        type (str): Type of the annotation,
        begin (int): Start index of annotated string,
        end (int): End index of annotated string,
        excerpt (str): Excerpt of annotated string
    """

    def __init__(
        self, file_name: str, id: str, type: str, begin: int, end: int, excerpt: str
    ) -> None:
        self.file_name = file_name
        self.id = id
        self.type = type
        self.begin = begin
        self.end = end
        self.excerpt = excerpt

    def to_string(self, usage: str = "annotation") -> str:
        """
        Create string with all attributes of the Annotation object.

        Arguments:
        ---------
            usage (str): Formats return string depending on usage.

        Returns:
        -------
            str: Annotation object as string.

        Raises:
        ------
            ValueError: If 'usage' is neither "annotation" nor "info".

        """
        match usage:
            case "info":
                string = f"Annotation({self.file_name}, {self.id}, {self.type}, {self.begin}, {self.end}, {self.excerpt})"
            case "annotation":
                string = (
                    f"{self.id}\t{self.type} {self.begin} {self.end}\t{self.excerpt}\n"
                )
            case _:
                raise ValueError("Invalid usage argument.")

        return string
