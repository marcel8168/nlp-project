import importlib
import os
import platform
import shutil
import subprocess
from typing import Optional

# import pyautogui
import constants


class System:
    """
    The System class manages system-related operations for the active learning toolset.
    This class provides methods for interacting with the operating system, managing constants,
    copying configuration files, starting and terminating Docker containers, and initiating the BRAT server.

    Attributes
    ----------
        operating_system (str): The name of the operating system (e.g., "Windows", "Linux").
        slash (str): The path separator used by the operating system.
    """
    def __init__(self) -> None:
        # Determine the operating system for path handling
        self.operating_system = platform.system()
        self.slash = "\\" if self.operating_system.lower() == "windows" else "/"
        pass
        
    def get_file_names_from_path(self, path_to_brat: str, folder_name: Optional[str] = None, collection_name: Optional[str] = None):
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
        collection_path = path_to_brat + self.slash + folder_name + self.slash if folder_name else path_to_brat
        collection_path += collection_name + self.slash if collection_name else ""
        file_names = os.listdir(collection_path) if os.path.isdir(collection_path) else []
        return collection_path, file_names
    
    def start_docker(self) -> None:
        """
        Starts the Docker container.

        Raises
        ------
            subprocess.CalledProcessError: If an error occurs while starting the Docker container.
        """
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
        """
        Terminates the Docker container.

        Raises
        ------
            subprocess.CalledProcessError: If an error occurs while terminating the Docker container.
        """
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

    def copy_config_directory(self) -> None:
        """
        Copies the configuration directory to the destination.

        Raises
        ------
            Exception: If an error occurs during the copying process.
        """
        source_dir = "./config/"
        destination_dir = "../brat/"

        try:
            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                destination_item = os.path.join(destination_dir, item)
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, destination_item)
            else:
                shutil.copy2(source_item, destination_item)
        except Exception as e:
            print(f"An error occurred while copying the directory: {e}")

    def start_brat(self) -> None:
        """
        Starts the BRAT server using the standalone.py script.

        Raises
        ------
            subprocess.CalledProcessError: If an error occurs while starting the BRAT server.
        """
        try:
            subprocess.Popen(["python", "standalone.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")