import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = pd.read_csv("../resources/CE802_Ass_2019_Data.csv")
features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19']

X = data.loc[:, features]
y = data.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree_clf = tree.DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
svm_clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
bayes_clf = GaussianNB().fit(X_train, y_train)

print("Tree: %s" % tree_clf.score(X_test, y_test))
print("SVM: %s" % svm_clf.score(X_test, y_test))
print("Naive Bayes: %s" % bayes_clf.score(X_test, y_test))
