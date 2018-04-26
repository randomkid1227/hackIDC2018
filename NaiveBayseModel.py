from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def NaiveBayseModel(i_features_train, i_target_train, i_features_test, i_target_test):

    i_features_train = preprocessing.scale(i_features_train)
    i_features_test = preprocessing.scale(i_features_test)
    model = GaussianNB()
    model.fit(i_features_train, i_target_train)
    prediction = model.predict(i_features_test)
    print(accuracy_score(i_target_test, prediction, normalize=True))




