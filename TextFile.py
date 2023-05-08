from typing import Optional

import pandas as pd
import nltk


class TextFile:
    """
    The AnnotationFile object contains information of an annotation file.

    Attributes
    ----------
        file_name (str): Name of the file (with file extension) where the annotation is stored in,
        path (str): Location where the file is stored at
    """

    def __init__(self, file_name: str, path: str, text: Optional[str] = None) -> None:
        self.file_name = file_name
        self.path = path
        self.text = text

        nltk.download("punkt")

    def read(self) -> str:
        """
        Read the content of the annotation file object.

        Returns
        -------
            set[Annotation]: Set of Annotation objects.
        """

        with open(self.path + self.file_name, "r", encoding="utf8") as file:
            self.text = file.read()
            self.text.replace("'", '"')

        return self.text
    
    def get_sentence_info(self) -> pd.DataFrame:
        if not self.text:
            self.read()
        text = self.text
        sentences = nltk.sent_tokenize(text=self.text)
        sentence_info = []
        end_index = 0

        for sentence in sentences:
            relative_index = text.find(sentence)
            start_index = end_index + relative_index
            if relative_index > -1:
                length = len(sentence)
                end_index = start_index + length
                sentence_info.append([sentence, start_index, end_index])
                text = text[relative_index + length:]
            else:
                raise Exception(f"Sentence '{sentence}' not found.")
            
        return pd.DataFrame(data=sentence_info, columns=["sentence", "start", "end"])
    