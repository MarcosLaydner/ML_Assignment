import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data = pd.read_csv("../resources/CE802_Ass_2019_Data - Adjusted.csv")
features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20']

X = data.loc[:, features]
y = data.Class

tree_clf = tree.DecisionTreeClassifier(random_state=0).fit(X, y)
svm_clf = svm.SVC(kernel='linear', C=1).fit(X, y)
bayes_clf = GaussianNB().fit(X, y)

print("Tree: %s" % cross_val_score(tree_clf, X, y, cv=10).mean())
print("SVM: %s" % cross_val_score(svm_clf, X, y, cv=10).mean())
print("Naive Bayes: %s" % cross_val_score(bayes_clf, X, y, cv=10).mean())


