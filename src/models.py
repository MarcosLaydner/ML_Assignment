import copy

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


def train_model(data, features, classifier):
    X = data.loc[:, features]
    y = data.Class

    clf = classifier.fit(X, y)

    return cross_val_score(clf, X, y, cv=10).mean()


def feature_selection(score, classifier):

    useful_features = copy.deepcopy(feature_list)
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


def model_comparison(dataset):
    dt_score = train_model(dataset, feature_list, tree.DecisionTreeClassifier(random_state=0))
    svm_score = train_model(dataset, feature_list, svm.SVC(kernel='rbf', C=1))
    nb_score = train_model(dataset, feature_list, GaussianNB())

    print(dt_score, svm_score, nb_score)

    print("DT: ")
    feature_selection(dt_score, tree.DecisionTreeClassifier(random_state=0))
    print("SVM: ")
    feature_selection(svm_score, svm.SVC(kernel='rbf', C=1))
    print("NB: ")
    feature_selection(nb_score, GaussianNB())


data = pd.read_csv("../resources/CE802_Ass_2019_Data.csv")
feature_list = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20']

imputed_data = IterativeImputer(max_iter=10, random_state=0).fit_transform(data.values)
data = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
print(data.values)

normalized_values = preprocessing.MinMaxScaler().fit_transform(data.values)
normalized_data = pd.DataFrame(normalized_values, columns=data.columns, index=data.index)

print("Not Normalized")
model_comparison(data)
print("\nNormalized")
model_comparison(normalized_data)




