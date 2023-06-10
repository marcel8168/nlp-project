from setuptools import setup, find_packages

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name='nlp_project',
    version='1.0', 
    author='Marcel Hiltner',
    author_email='marcel.hiltner@fau.de',
    url="https://github.com/marcel8168/nlp-project",
    packages=find_packages(),
    description="NLP-Project @FAU",
    long_description=long_description,
    install_requires=[
        'datasets==2.12.0',
        'evaluate==0.4.0',
        'modAL==0.49.2348',
        'nltk==3.8.1',
        'numpy==1.23.5',
        'pandas==2.0.1',
        'scikit_learn==1.2.2',
        'setuptools==67.8.0',
        'torch==2.0.1',
        'transformers==4.29.1'
    ]
)
