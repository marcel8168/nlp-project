import importlib
import os
import platform
import stat
import subprocess
from typing import Optional, Tuple

import pyautogui
import constants


class System:
    def __init__(self) -> None:
        self.operating_system = platform.system()
        pass
        
    def get_file_names_from_path(self, path_to_brat: str, folder_name: Optional[str] = None, collection_name: Optional[str] = None) -> Tuple[str, list[str]]:
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
        slash = "\\" if self.operating_system.lower() == "windows" else "/"
        collection_path = path_to_brat + slash + folder_name + slash if folder_name else path_to_brat
        collection_path += collection_name + slash if collection_name else ""
        file_names = os.listdir(collection_path) if os.path.isdir(collection_path) else []
        return collection_path, file_names
    
    def start_docker(self) -> None:
        try:
            if self.operating_system.lower() == "windows":
                subprocess.Popen(['start', 'cmd', '/c', 'docker compose up'], shell=True)
            elif self.operating_system.lower() == "linux":
                subprocess.Popen(['docker', 'compose', 'up'])
            elif self.operating_system.lower() == "darwin":
                subprocess.run(['osascript', '-e', 'tell app "Terminal" to do script "docker compose up"'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

    def terminate_docker(self) -> None:
        try:
            if self.operating_system.lower() == "windows":
                subprocess.Popen(['start', 'cmd', '/c', 'docker compose down'], shell=True)
            elif self.operating_system.lower() == "linux":
                subprocess.Popen(['docker', 'compose', 'down'])
            elif self.operating_system.lower() == "darwin":
                subprocess.run(['osascript', '-e', 'tell app "Terminal" to do script "docker compose down"'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

    def get_constant(self, constant_name: str) -> str:
        """
        Retrieve the value of a constant from the `constants.py` file.

        Arguments
        ---------
            constant_name (str): The name of the constant.

        Returns
        -------
            str: The value of the constant.

        Raises
        ------
            AttributeError: If the constant does not exist in `constants.py`.
        """
        spec = importlib.util.spec_from_file_location("constants", "constants.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, constant_name):
            constant_value = getattr(module, constant_name)
            return constant_value
        else:
            raise AttributeError(f"Constant '{constant_name}' does not exist in constants.py")

    def set_constant_value(self, constant_name, value) -> None:
        """
        Save a value into a constant in the `constants.py` file.

        Arguments
        ---------
            constant_name (str): The name of the constant.
            value (object): The value to be assigned to the constant.

        Raises
        ------
            AttributeError: If the constant does not exist in `constants.py`.
        """
        if hasattr(constants, constant_name):
            setattr(constants, constant_name, value)
        else:
            raise AttributeError(f"Constant '{constant_name}' does not exist in constants.py")

        # Save the updated module to the file
        with open('constants.py', 'w') as file:
            for name, val in constants.__dict__.items():
                if not name.startswith('__'):
                    file.write(f"{name} = {repr(val)}\n")

    def reload(self) -> None:
        pyautogui.hotkey('F5')

    def give_Permissions(self, file_path: str) -> None:
        if self.operating_system.lower() == "linux":
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        elif self.operating_system.lower() == "darwin":
            os.chmod(file_path, 0o666)