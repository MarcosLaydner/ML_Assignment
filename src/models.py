import copy

import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


def train_model(data, features, classifier):
    X = data.loc[:, features]
    y = data.Class

    clf = classifier.fit(X, y)

    return cross_val_score(clf, X, y, cv=10).mean()


def feature_selection(score, classifier):

    useful_features = copy.deepcopy(feature_list)
    print(useful_features)
    max_score = score

    for feature in feature_list:
        useful_features.remove(feature)
        list = copy.deepcopy(useful_features)
        tmp_score = train_model(data, list, classifier)
        if tmp_score > score:
            max_score = tmp_score
        else:
            useful_features.append(feature)

    print(useful_features, max_score)
    return useful_features


data = pd.read_csv("../resources/CE802_Ass_2019_Data - Adjusted.csv")
feature_list = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20']

# print("Decision Tree: %s" % train_model(data, features, tree.DecisionTreeClassifier(random_state=0)))
# print("SVM : %s" % train_model(data, features, svm.SVC(kernel='linear', C=1)))
# print("Naive Bayes: %s" % train_model(data, features, GaussianNB()))

dt_score = train_model(data, feature_list, tree.DecisionTreeClassifier(random_state=0))
svm_score = train_model(data, feature_list, svm.SVC(kernel='linear', C=1))
nb_score = train_model(data, feature_list, GaussianNB())

print(dt_score)
print(svm_score)
print(nb_score)

print("DT: ")
feature_selection(dt_score, tree.DecisionTreeClassifier(random_state=0))
print("SVM: ")
feature_selection(svm_score, svm.SVC(kernel='linear', C=1))
print("NB: ")
feature_selection(nb_score, svm.SVC(kernel='linear', C=1))

# def feature_selection(score, classifier):
#
#     useful_features = copy.deepcopy(feature_list)
#     print(useful_features)
#     max_score = score
#
#     for feature in useful_features:
#         useful_features.remove(feature)
#         tmp_score = train_model(data, useful_features, classifier)
#         if tmp_score > score:
#             useful_features.append(useful_features)
#         else:
#             max_score = tmp_score
#
#     print(useful_features, max_score)
#     return useful_features
