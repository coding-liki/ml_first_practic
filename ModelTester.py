import numpy as np

from typing_extensions import Protocol

from sklearn.metrics import classification_report

from DataFetcher import Data


class Model(Protocol):
    def fit(self, x, y):
        pass

    def predict(self, x) -> np.ndarray:
        pass


class ModelTester:
    model: Model = None

    def __init__(self, model: Model):
        self.model = model

    def test(self, data: Data):
        self.model.fit(data.train_features, data.train_labels)
        print(len(data.test_features))

        test_result = self.model.predict(data.test_features)
        print(len(test_result))
        target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

        print(classification_report(data.test_labels, test_result, target_names=target_names))
        pass
