from LinearSvmModel import LinearSvmModel
from NaiveBayseModel import NaiveBayseModel
from RandomForest import RandomForest
from reader import DatasetFactory


def Main():
    fileNameArray = [
        {
            "path": "RawData\\OpenBCI-RAW-k2_with_class.csv", # filename
            "with_vector": False, # include or exclude x,y,z vector
            "time_to_remove_from_start": "0.0", # In seconds
            "time_to_remove_from_end": "1.0" # In seconds
        }
    ]
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

