from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score


def LinearSvmModel(i_features_train, i_target_train, i_features_test, i_target_test):

    C = 1.0
    model = svm.LinearSVC(C=C).fit(i_features_train, i_target_train)
    prediction = model.predict(i_features_test)
    print(accuracy_score(i_target_test, prediction, normalize=True))


