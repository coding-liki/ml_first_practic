from DataFetcher import DataFetcher
from ModelTester import ModelTester
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
data_fetcher = DataFetcher("./data/train.csv", "./data/test.csv")

ModelTester(model).test(data_fetcher.fetch())





