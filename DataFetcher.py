import numpy as np
import pandas as pd

label_map: dict[str, int] = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2, 'SITTING': 3, 'STANDING': 4,
                             'LAYING': 5}


def read_csv(path: str):
    return pd.read_csv(path)


class Data:
    feature_names: list = []
    train_features: np.ndarray = []
    train_labels: list = []

    test_features: np.ndarray = []
    test_labels: list = []

    def __init__(
            self,
            train_features: np.ndarray,
            train_labels: list,
            test_features: np.ndarray,
            test_labels: list,
            feature_names: list
    ):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.feature_names = feature_names


class DataFetcher:
    data: Data = None
    train_path: str = None
    test_path: str = None

    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def fetch(self) -> Data:
        if self.data is None:
            self.read_data()

        return self.data

    def read_data(self):
        train_data = read_csv(self.train_path)
        train_features = train_data.values[:, :-2]
        train_labels = train_data['Activity'].map(label_map).values

        test_data = read_csv(self.test_path)

        test_features = test_data.values[:, :-2]
        test_labels = test_data['Activity'].map(label_map).values

        self.data = Data(train_features, train_labels, test_features, test_labels, test_data.columns[:-2])
        pass
