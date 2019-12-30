import pandas as pd
from sklearn import svm
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import cross_val_score

data = pd.read_csv("../resources/CE802_Ass_2019_Data - Adjusted.csv")
feature_list = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20']

X = data.loc[:, feature_list]
y = data.Class
clf = svm.SVC(kernel='rbf', C=1)

print(cross_val_score(clf.fit(X, y), X, y, cv=10).mean())

# rfecv = RFECV(estimator=svm.SVC(kernel='linear', C=1), step=1, cv=StratifiedKFold(10), scoring='accuracy')
# rfecv.fit(X, y)
#
# print(rfecv.grid_scores_)

# selector = SelectFromModel(estimator=svm.SVC(kernel='linear', C=1)).fit(X, y)
#
#
# print(cross_val_score(svm.SVC(kernel='rbf', C=1).fit(selector.transform(X), y), selector.transform(X), y, cv=10).mean())

X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
clf1 = svm.SVC(kernel='rbf', C=1)
print((cross_val_score(clf1.fit(X_new, y), X_new, y, cv=10)).mean())
