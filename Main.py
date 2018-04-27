from LinearSvmModel import LinearSvmModel
from NaiveBayseModel import NaiveBayseModel
from RandomForest import RandomForest
from reader import DatasetFactory


def Main():
    fileNameArray6 = [
        {
            "path": "RawData\\OpenBCI-RAW-k2_with_class.csv", # filename
            "with_vector": False, # include or exclude x,y,z vector
            "time_to_remove_from_start": "0.0", # In seconds
            "time_to_remove_from_end": "1.0", # In seconds
            "enable": False
        }
    ]
    fileNameArray = [{"path": "RawData\\OpenBCI-RAW-k1_minus5sec_with_class.csv", "with_vector": False, "time_to_remove_from_start": "0.0",
                      "time_to_remove_from_end": "0.0", "enable": False}]
    dataSet = DatasetFactory(fileNameArray)
    X_train = dataSet.data[0].X_train
    X_test = dataSet.data[0].X_test
    y_train = dataSet.data[0].y_train
    y_test = dataSet.data[0].y_test
    print(X_train)
    print("------------------------------------")
    print(X_test)
    NaiveBayseModel(X_train, y_train, X_test, y_test)
    LinearSvmModel(X_train, y_train, X_test, y_test)
    RandomForest(X_train, y_train, X_test, y_test)


if '__main__' == __name__:
        Main()

