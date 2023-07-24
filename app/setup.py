from setuptools import find_packages
from distutils.core import setup

setup(
    name='BRAT_AL_TOOL',
    version='1.0', 
    author='Marcel Hiltner',
    author_email='marcel.hiltner@fau.de',
    url="https://github.com/marcel8168/nlp-project",
    packages=find_packages(),
    description="Toolset for active learning based control of medical free-text annotations",
    install_requires=[
        'datasets==2.12.0',
        'evaluate==0.4.0',
        'modAL-python==0.4.2.1',
        'nltk==3.8.1',
        'numpy==1.23.5',
        'pandas==2.0.1',
        'scikit_learn==1.2.2',
        'setuptools==67.8.0',
        'torch==2.0.1',
        'transformers==4.29.1',
        'seqeval==1.2.2',
        'accelerate==0.20.3',
        'pyautogui==0.9.54',
        'matplotlib==3.7.2'
    ]
)
