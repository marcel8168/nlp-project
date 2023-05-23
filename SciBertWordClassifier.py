from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.base import BaseEstimator
import torch
import numpy as np


class SciBertWordClassifier(BaseEstimator):
    """
    A scikit-learn compatible classifier that uses the SciBERT model for word-level sequence classification.
    """

    def __init__(self, num_classes: int):
        """
        Initializes the SciBertWordClassifier.

        Args:
            num_classes (int): The number of classes for classification.
        """
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=num_classes)

    def _encode_text_to_tensors(self, X, tokenizer, max_length):
        input_ids = []
        attention_masks = []

        for word in X:
            encoded = tokenizer.encode_plus(
                word,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def fit(self, X, y) -> BaseEstimator:
        """
        Fits the SciBertWordClassifier to the provided training data.

        Args:
            X (array-like of shape (n_samples,)): The input words.
            y (array-like of shape (n_samples,)): The target values.

        Returns:
            self
        """
        input_ids, attention_masks = self._encode_text_to_tensors(X=X, tokenizer=self.tokenizer, max_length=16)

        labels = torch.tensor(y)

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        for epoch in range(3):  # Fine-tune for 3 epochs
            for batch in dataloader:
                optimizer.zero_grad()

                input_ids = batch[0]
                attention_masks = batch[1]
                labels = batch[2]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)

                loss = outputs.loss
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Predicts class probabilities for the input words.

        Args:
            X (array-like of shape (n_samples,)): The input words.

        Returns:
            list of shape (n_samples, n_classes): The class probabilities of the input words.
        """
        input_ids, attention_masks = self._encode_text_to_tensors(X=X, tokenizer=self.tokenizer, max_length=16)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_masks
            )

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.numpy()

    def predict(self, X) -> np.ndarray:
        """
        Predicts the class labels for the input words.

        Args:
            X (array-like of shape (n_samples,)): The input words.

        Returns:
            numpy.ndarray of shape (n_samples,): The predicted class labels.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)+-1
