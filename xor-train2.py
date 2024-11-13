from sklearn import svm, metrics
import pandas as pd


xor_input = [
    #P, Q, result
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

xor_df = pd.DataFrame(xor_input)
xor_data = xor_df.loc[:, 0:1]
xor_label = xor_df.loc[:,2]



clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)


ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률== ", ac_score)