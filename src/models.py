import copy

import pandas as pd
from numpy import shape
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.impute._iterative import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# Trains a model and returns it with its cross validation score
def train_model(data, features, classifier):
    X = data.loc[:, features]
    y = data.Class

    clf = classifier.fit(X, y)

    return {'model': clf,
            'score': cross_val_score(clf, X, y, cv=10).mean()}


# returns the accuracy of a model with a given dataset
def test_model(dataset, features, classifier):
    X = data.loc[:, features]
    y = data.Class

    return classifier.score(X, y)


# Removes features and checks the new accuracy score of the model. If it improves, that feature is excluded.
# Returns a list with the features that weren't removed
def feature_selection(dataset, score, classifier):

    useful_features = copy.deepcopy(feature_list)
    max_score = score

    for feature in feature_list:
        useful_features.remove(feature)
        list = copy.deepcopy(useful_features)
        tmp_score = train_model(dataset, list, classifier)['score']
        if tmp_score > score:
            max_score = tmp_score
        else:
            useful_features.append(feature)

    print(useful_features, max_score)
    return useful_features


# Prints the accuracy of each of the 3 chosen models, before and after feature selection, for comparison
def model_comparison(dataset):
    dt_score = train_model(dataset, feature_list, tree.DecisionTreeClassifier(random_state=0))['score']
    svm_score = train_model(dataset, feature_list, svm.SVC(kernel='linear', C=1))['score']
    nb_score = train_model(dataset, feature_list, GaussianNB())['score']

    print(dt_score, svm_score, nb_score)

    print("DT: ")
    feature_selection(dataset, dt_score, tree.DecisionTreeClassifier(random_state=0))
    print("SVM: ")
    feature_selection(dataset, svm_score, svm.SVC(kernel='linear', C=1))
    print("NB: ")
    feature_selection(dataset, nb_score, GaussianNB())


data = pd.read_csv("../resources/CE802_Ass_2019_Data.csv")
feature_list = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20']

# Mean imputation
# imputed_data = SimpleImputer(strategy='mean').fit_transform(data.values)

# Regression imputation
imputed_data = IterativeImputer(max_iter=10, skip_complete=True).fit_transform(data.values)
data = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)

# Splits data in train and test data
split = train_test_split(data, test_size=0.20)
data = split[0]
test_data = split[1]

# data normalization
normalized_values = preprocessing.MinMaxScaler().fit_transform(data.values)
normalized_data = pd.DataFrame(normalized_values, columns=data.columns, index=data.index)

# print("Not Normalized")
# model_comparison(data)
# print("\nNormalized")
# model_comparison(normalized_data)

# test actual performance of the best model, and compares it with the training performance
model = train_model(data, feature_list, svm.SVC(kernel='linear', C=1))
print(model['score'], test_model(test_data, feature_list, model['model']))

# Gets test dataset provided, and imputes missing values
test_dataset = pd.read_csv("../resources/CE802_Ass_2019_Test.csv")
test_dataset = test_dataset.drop(columns=['Class'])
regression_imputation = IterativeImputer(max_iter=10, skip_complete=True).fit_transform(test_dataset.values)
test_dataset = pd.DataFrame(regression_imputation, columns=test_dataset.columns, index=test_dataset.index)
print(test_dataset.head(10))

# Uses the chosen model for predicting the new test data
predictions = model['model'].predict(test_dataset)
test_dataset['Class'] = predictions

print(test_dataset.head(10))

# Writes results to a csv
test_dataset.to_csv('../resources/CE802_Ass_2019_Test_Result.csv')





