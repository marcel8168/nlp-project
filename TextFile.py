import platform
import re
from typing import Iterable, Optional
import io
import pandas as pd
import nltk


class TextFile:
    """
    The TextFile object contains information of an text file.

    Attributes
    ----------
        file_name (str): Name of the file (with file extension) where the text is stored in,
        path (str): Location where the file is stored at,
        text (str): Content of the file
    """

    def __init__(self, file_name: str, path: str, text: Optional[str] = None) -> None:
        self.file_name = file_name
        self.path = path
        self.text = text

        nltk.download("punkt", quiet=True)

    def read(self) -> str:
        """
        Read the content of the text file object.

        Returns
        -------
            str: Content of the text file.
        """
        operating_system = platform.system()
        slash = "\\" if operating_system == "Windows" else "/"
        full_path = self.path + slash + self.file_name

        with io.open(full_path, "r", encoding="utf8", newline='') as file:
            self.text = file.read()
            self.text.replace("'", '"')

        return self.text
    
    def get_sentence_info(self) -> pd.DataFrame:
        """
        Get all sentences with start and end index information.

        Raises
        ------
            Exception: If tokenized sentence not found in text.

        Returns
        -------
            Pandas.DataFrame[str: sentence, int: start, int: end]: DataFrame with all sentences with information about start and end index.

        """
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
    
    def contains(self, excerpts: Iterable[str]) -> list:
        """
        Get all occurrences of given excerpts in the TextFile object.

        Arguments
        ---------
            excerpts (Iterable[str]): Excerpts that should be found in the TextFile object.

        Returns
        -------
            list[str, int, int]: List of found excerpts with start index and end index.

        """
        if not self.text:
            self.read()
            
        excerpt_infos = set()
        for excerpt in excerpts:
            pattern = r"\b{}\b".format(excerpt)
            matches = re.finditer(pattern, self.text, re.IGNORECASE)
            for match in matches:
                excerpt_infos.add((match.group(0), match.start(), match.end()))
        
        return list(excerpt_infos)
    