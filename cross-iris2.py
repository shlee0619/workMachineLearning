from sklearn import svm, metrics, model_selection
from sklearn.model_selection import KFold
import random, re
import pandas as pd

csv = pd.read_csv('iris.csv')

data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

clf = svm.SVC()
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print("각각의 정답률 =", scores)
print("평균정답률=", scores.mean())