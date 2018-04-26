from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def RandomForest(i_features_train, i_target_train, i_features_test, i_target_test):
    model = RandomForestClassifier(n_estimators=1000, random_state=1234)
    model.fit(i_target_train, i_target_test)
    predicted = model.predict(i_features_train)
    print(accuracy_score(i_features_test, predicted))
