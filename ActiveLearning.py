from typing import Union
from modAL.uncertainty import uncertainty_sampling
import sklearn
import numpy as np
import scipy

class ActiveLearning:
    def __init__(self) -> None:
        pass

    def iteration(self, classifier: sklearn.base.BaseEstimator,
                  unlabeled_data: Union[list, np.ndarray],
                  num_to_annotate: int = 1):
        indices = uncertainty_sampling(classifier=classifier, X=unlabeled_data, n_instances=num_to_annotate)
        
        if isinstance(unlabeled_data, list):
            unlabeled_data = np.array(unlabeled_data)
        uncertain_samples = list(unlabeled_data[indices])

        return uncertain_samples