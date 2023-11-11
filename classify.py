import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from DataFetcher import DataFetcher
from ModelTester import ModelTester
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from math import sqrt

data_fetcher = DataFetcher("./data/train.csv", "./data/test.csv")


def test_k_neighbours():
    print("Testing K-Nearest Neighbours")

    algorithms_to_test = {'auto', 'ball_tree', 'kd_tree', 'brute'}
    n_neighbors_to_test = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, round(sqrt(len(data_fetcher.fetch().test_labels)) / 2),
                           round(sqrt(len(data_fetcher.fetch().test_labels)))}
    weights_to_test = {'uniform', 'distance'}
    p_to_test = {1, 2}

    for algorithm in algorithms_to_test:
        for weights in weights_to_test:
            for p in p_to_test:
                for n_neighbors in n_neighbors_to_test:
                    model = KNeighborsClassifier(
                        algorithm=algorithm,
                        n_neighbors=n_neighbors,
                        weights=weights,
                        p=p,
                        n_jobs=-1
                    )

                    ModelTester(model).test(data_fetcher.fetch())


test_k_neighbours()


def test_svm():
    print("Testing Support Vectors Method")

    c_to_test = {0.1, 0.3, 0.6, 1, 1.3, 2, 4}
    decision_function_shape_to_test = {'ovo', 'ovr'}
    kernels_to_test = {'linear', 'poly', 'rbf', 'sigmoid'}
    for decision_function_shape in decision_function_shape_to_test:
        for kernel in kernels_to_test:
            for c in c_to_test:
                model = svm.SVC(
                    C=c,
                    decision_function_shape=decision_function_shape,
                    kernel=kernel
                )

                ModelTester(model).test(data_fetcher.fetch())


test_svm()


def test_decision_tree():
    print("Testing Decision Trees")
    criterion_to_test = {'gini', 'entropy', 'log_loss'}
    splitter_to_test = {'best', 'random'}
    ccp_alpha_to_test = {0.0, 0.05, 0.1, 0.2}

    for criterion in criterion_to_test:
        for splitter in splitter_to_test:
            for ccp_alpha in ccp_alpha_to_test:
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    splitter=splitter,
                    ccp_alpha=ccp_alpha,
                )

                ModelTester(model).test(data_fetcher.fetch())

                importance_data = pd.DataFrame(model.feature_importances_, data_fetcher.fetch().feature_names)
                print(importance_data.sort_values(by=0, ascending=False).head(10))


test_decision_tree()
