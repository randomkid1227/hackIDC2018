from LinearSvmModel import LinearSvmModel
from NaiveBayseModel import NaiveBayseModel
from RandomForest import RandomForest
from reader import DatasetFactory


def Main():


    fileNameArray = ["ice (1).csv"]
    dataSet = DatasetFactory(fileNameArray)
    X_train = dataSet.data[0].X_train
    X_test = dataSet.data[0].X_test
    y_train = dataSet.data[0].y_train
    y_test = dataSet.data[0].y_test

    NaiveBayseModel(X_train, y_train, X_test, y_test)
    LinearSvmModel(X_train, X_test, y_train, y_test)
    RandomForest(X_train, X_test, y_train, y_test)


if '__main__' == __name__:
        Main()

