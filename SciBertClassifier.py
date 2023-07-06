import os
import statistics
from typing import Iterable, Optional
import evaluate
from joblib import dump, load
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, pipeline
from sklearn.base import BaseEstimator
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt

from constants import METRICS_FILE_PATH, TARGET_CLASS


class SciBertClassifier(BaseEstimator):
    """
    A scikit-learn compatible classifier that uses the SciBERT model for token classification.

    Arguments
    ---------
        label (str): The label for the target class.
        label_list (List[str]): The list of all possible labels.
        num_classes (int): The number of classes for classification.
        path (str, optional): The path to save/load the model. Defaults to None.
        learning_rate (float, optional): The learning rate for model training. Defaults to 5e-5.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        num_epochs (int, optional): The number of training epochs. Defaults to 3.
        weight_decay (float, optional): The weight decay for training. Defaults to 0.01.
        logging_steps (int, optional): The frequency of logging during training. Defaults to 100.
        token_aggregation (str, optional): Aggregation method for calculating the probability of split words.

    Public Variables
    ----------------
        metric (object): An evaluation metric object for sequence labeling, loaded from the "seqeval" library.
        model_checkpoint (str): The checkpoint name or path of the SciBERT model.
        model_name (str): The name of the loaded SciBERT model.
        task (str): The task type, set to "ner" for named entity recognition.
        label (str): The label for the target class.
        label_list (List[str]): The list of all possible labels.
        tokenizer (object): The tokenizer object for the SciBERT model.
        model (object): The token classification model based on the SciBERT architecture.
        path (str): The path to save/load the model.
        args (object): Training arguments for fine-tuning the model.
        data_collator (object): Data collator object for token classification.

    Methods
    -------
        fit(X) -> BaseEstimator:
        Trains the model on the given input and target data.

        predict(X) -> Iterable:
            Predicts the class labels for the input text.

        predict_proba(X) -> Iterable:
            Predicts the class probabilities for the input text.

        generate_row_labels(text: dict) -> dict:
            Generates row labels (token labels) for a given text.

        compute_metrics(p) -> dict:
            Computes evaluation metrics based on the predictions and labels.

        whole_word_prediction(input: list, aggregation_strategy: str = "max") -> list:
            Perform whole word prediction by aggregating probabilities for split-up words.

        save(path: Optional[str]):
            Saves the model to the specified path.

        load(path: Optional[str]):
            Loads a saved model from the specified path.
    """

    def __init__(self, num_classes: int, label, label_list, batch_size=16, learning_rate=1e-5, num_epochs=5, weight_decay=0.05, logging_steps=1, path="", token_aggregation=""):
        self.metric = evaluate.load("seqeval")
        
        self.model_checkpoint = "allenai/scibert_scivocab_uncased"
        self.model_name = self.model_checkpoint.split("/")[-1]
        self.task = "ner"

        self.label = label
        self.label_list = label_list

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, num_labels=num_classes)

        self.path = path if path else "./model/SciBertClassifier.joblib"

        self.token_aggregation = token_aggregation

        self.args = TrainingArguments(
            f"{self.model_name}-finetuned-{self.task}",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_steps=logging_steps
        )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    def load(self, path: Optional[str] = "") -> None:
        """
        Loads a saved SciBertClassifier model from the specified path.

        Arguments
        ---------
            path (str, optional): The path to the saved model file. If not provided, uses the default path.

        Returns
        -------
            BaseEstimator: The loaded BaseEstimator model.
        """
        path = path if path else self.path
        self.model = load(path)
    
    def save(self, path: Optional[str] = "") -> None:
        """
        Saves the SciBertClassifier model to the specified path.

        Parameters
        ----------
            path (str, optional): The path to save the model file. If not provided, uses the default path.
        """
        path = path if path else self.path
        dir = os.path.dirname(path)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dump(self.model, path)

    def generate_row_labels(self, text: dict) -> dict:
        """
        Generates row labels (token labels) for a given text.

        Arguments
        ---------
            text (dict): The input text containing 'text', 'drug_indices_start', and 'drug_indices_end' fields.

        Returns
        -------
            dict: A dictionary containing the token labels for each input token, appended to the 'labels' field.
        """
        labels = []
        lbl = "O"
        prefix = ""
        index = 0

        tokens = self.tokenizer(text["text"], return_offsets_mapping=True)

        for n in range(len(tokens["input_ids"])):
            offset_start, offset_end = tokens["offset_mapping"][n]

            # should only happen for [CLS] and [SEP]
            if offset_end - offset_start == 0:
                labels.append(-100)
                continue
            
            if index < len(text["drug_indices_start"]) and offset_start == text["drug_indices_start"][index]:
                lbl = self.label
                prefix = "B-"
            
            labels.append(self.label_list.index(f"{prefix}{lbl}"))
                
            if index < len(text["drug_indices_end"]) and offset_end == text["drug_indices_end"][index]:
                lbl = "O"
                prefix = ""
                index += 1

            # need to transition "inside" if we just entered an entity
            if prefix == "B-":
                prefix = "I-"
        
        tokens["labels"] = labels
        
        return tokens

    def fit(self, X: Dataset) -> BaseEstimator:
        """
        Fits the model to the training data and performs training.

        Arguments
        ---------
            X (dict): The labeled input data, containing training and test datasets.

        Returns
        -------
            BaseEstimator: The fitted estimator object.

        """
        trainer = Trainer(
            self.model,
            self.args,
            train_dataset=X["train"],
            eval_dataset=X["test"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,

        )

        trainer.train()

        return self

    def predict_proba(self, X) -> Iterable:
        """
        Predicts the class probabilities for the input words.

        Arguments
        ---------
            X (array-like of shape (n_samples,)): The input words.

        Returns
        -------
            numpy.ndarray of shape (n_samples,): The predicted class probabilities.
        """
        predictions = self.predict(X=X)
        predictions = np.array([[data["score"] for data in dataset] for dataset in predictions]).flatten()
        counter_predictions = 1 - predictions.copy()
        
        return np.c_[predictions, counter_predictions]

    

    def predict(self, X) -> Iterable:
        """
        Predicts the class labels for the input text.

        Arguments
        ---------
            X (array-like of shape (n_samples,)): The input words.

        Returns
        -------
            numpy.ndarray of shape (n_samples,): The predicted class labels.
        """
        ner_pipeline = pipeline(task="ner", model=self.model, tokenizer=self.tokenizer, device=-1)
        predictions = [self.whole_word_prediction(input=ner_pipeline(text), aggregation_strategy="max") for text in X]
        self.predictions = np.array([[word for word in prediction if word["word"].isalnum()] for prediction in predictions])

        return self.predictions

    def whole_word_prediction(self, input: list, aggregation_strategy: str = "max") -> list:
        """
        Perform whole word prediction by aggregating probabilities for split-up words.

        Arguments
        ---------
            input (list): List of dictionaries containing token probabilities and words.
            aggregation_strategy (str): Aggregation strategy for combining probabilities of split-up words.
                Supported values: "max" (maximum), "avg" (average). Default: "max".

        Returns
        -------
            list: List of dictionaries containing the aggregated probabilities and information for each whole word.
        """
        probabilities = []
        whole_word = ""
        whole_word_probabilities = []
        index = -1

        if self.token_aggregation:
            aggregation_strategy = aggregation_strategy if aggregation_strategy != "max" else self.token_aggregation
    
        for data in input:
            token_probability = data["score"]
            token_word = data["word"]
            if token_word.startswith("##"):
                whole_word += token_word[2:]
                whole_word_probabilities.append(token_probability)
            else:
                index += 1
                if len(whole_word_probabilities) > 1:
                    if aggregation_strategy == "max":
                        probabilities[-1]["score"] = max(whole_word_probabilities)
                    elif aggregation_strategy == "avg":
                        probabilities[-1]["score"] = statistics.mean(whole_word_probabilities)
                    probabilities[-1]["end"] = probabilities[-1]["start"] + len(whole_word)
                    probabilities[-1]["word"] = whole_word
            
                whole_word = data["word"]
                whole_word_probabilities = [token_probability]
                probabilities.append(data)
                probabilities[-1]["index"] = index

        return probabilities
    
    def performance_report(self, path_to_test_set: str):
        predictions = []
        ground_truth = []

        dataset = load_dataset("json", data_files=path_to_test_set)["train"]
        prediction_results = self.predict(dataset["text"])
        for idx, sentence in enumerate(prediction_results):
            for word in sentence:
                word_instance = word["word"].lower()
                predictions.append((word_instance, word["entity"]))
                if word["word"].lower() in [element.lower() for element in dataset[idx][TARGET_CLASS]]:
                    ground_truth.append((word_instance, "LABEL_1"))
                else:
                    ground_truth.append((word_instance, "LABEL_0"))

        metrics = self.compute_test_metrics(predictions, ground_truth)

        return metrics

    def compute_test_metrics(self, predictions, ground_truth):
        """
        Computes evaluation metrics for drug name detection.

        Arguments:
        - predictions (list): List of predicted drug names.
        - ground_truth (list): List of ground truth drug information.

        Returns:
        - metrics (dict): Dictionary containing computed evaluation metrics.
        """
        pred_set = set(predictions)
        gt_set = set(ground_truth)

        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(METRICS_FILE_PATH, mode='a', header=not os.path.exists(METRICS_FILE_PATH), index=False)

        metrics_df = pd.read_csv(METRICS_FILE_PATH)

        precision_list = metrics_df['precision']
        recall_list = metrics_df['recall']
        f1_list = metrics_df['f1']
        accuracy_list = metrics_df['accuracy']

        plt.plot(metrics_df.index, precision_list, label='Precision')
        plt.plot(metrics_df.index, recall_list, label='Recall')
        plt.plot(metrics_df.index, f1_list, label='F1-Score')
        plt.plot(metrics_df.index, accuracy_list, label='Accuracy')

        # Set labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('Training Metrics')

        # Add legend
        plt.legend()

        # Show the chart
        plt.show()

        return metrics
