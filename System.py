import os
import platform
from typing import Optional, Tuple


class System:
    def __init__(self) -> None:
        pass
        
    def get_file_names_from_path(self, path_to_brat: str, folder_name: str, collection_name: Optional[str]) -> Tuple[str, list[str]]:
        """
        Retrieves the file names from a specified path in the file system.

        Arguments
        ---------
            path_to_brat (str): The base path to the BRAT directory.
            folder_name (str): The name of the folder within the BRAT directory.
            collection_name (str): The name of the collection within the folder (optional).

        Returns
        -------
            Tuple[str, List[str]]: A tuple containing the collection path and a list of file names.
                                The collection path is formed by concatenating the base path, folder name,
                                and collection name (if provided). The list of file names contains the names
                                of the files within the collection path.
        """
        operating_system = platform.system()
        slash = "\\" if operating_system == "Windows" else "/"
        collection_path = path_to_brat + slash + folder_name + slash
        collection_path += collection_name + slash if collection_name else ""

        return collection_path, os.listdir(collection_path)
