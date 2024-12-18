from sklearn import svm, metrics
import pandas as pd
from sklearn.model_selection import train_test_split

#붓꽃의 csv데이터 읽어 들이기

csv = pd.read_csv('./붓꽃이미지jpg/iris.csv')
csv_data = csv[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
csv_label = csv["variety"]

train_data, test_data, train_label, test_label = \
    train_test_split(csv_data, csv_label)




clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print('정답률 = ', ac_score)
