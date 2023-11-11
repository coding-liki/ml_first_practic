import numpy as np

from typing_extensions import Protocol

from sklearn.metrics import classification_report

from DataFetcher import Data


class Model(Protocol):
    def fit(self, x, y):
        pass

    def predict(self, x) -> np.ndarray:
        pass

    def get_params(self, deep=True) -> dict:
        pass


class ModelTester:
    model: Model = None

    def __init__(self, model: Model):
        self.model = model

    def test(self, data: Data):
        print(self.model.get_params())
        self.model.fit(data.train_features, data.train_labels)

        test_result = self.model.predict(data.test_features)
        target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

        print(classification_report(data.test_labels, test_result, target_names=target_names, zero_division=0))
        pass
