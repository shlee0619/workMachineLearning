import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import matplotlib as mpl 
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False


import pandas as pd
import seaborn as sns
import numpy as np
#---------------------------------------------------------------------------------------------

# 순서1  Social_Network_Ads.csv 파일 읽기  400행 * 5열 
df = pd.read_csv('./data/Social_Network_Ads.csv')
print(df) #400행 * 5열 

# 순서2 x,y추출  UserID  Gender  Age  EstimatedSalary  Purchased 
x = df.loc[ : , 'Age':'EstimatedSalary'] 
y = df['Purchased'] 
print(x)

# 순서3 전처리과정 스케일링 조정  MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler_x  = MinMaxScaler()
x = scaler_x.fit_transform(x)


# 순서4 train_test_split(x,y,test_size) 확인  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1)    

 
# 순서5 분류에 관련 모델  구매여부(1구매, 0비구매) LogisticRegression
from sklearn.linear_model import  LogisticRegression
model  = LogisticRegression()

#순서6 학습훈련fit(), 예측predict() 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# print('y_pred')
# print(y_pred)

#순서7 기타 몇가지 확인
# print(x_train.shape) #(300, 2)
# print(x_test.shape)  #(100, 2)
print()

#순서8  수치화 
from sklearn.metrics  import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)
# print(cm)

# print(cm.sum()) #100 = 52+6+14+28 
print('수제 정확도',(52+28)/cm.sum()) 


from sklearn.metrics import accuracy_score
ss = accuracy_score(y_test, y_pred)
print('공식 정확도' , ss)

#기계학습, 통계함수, 수학함수 
from sklearn.metrics import classification_report
print('LogisticRegression 평가지표 보고서')
print(classification_report(y_test, y_pred))
print()
print()
print('- ' * 50)

from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
#                        np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

plt.figure(figsize=[10,7])
# plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  c=ListedColormap(('red', 'blue'))(i), label = j)
plt.title('LogisticRegression 분류  예측 결과 (test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
print()


# 새로운모델 KNeighborsClassifier
# 순서1 
df = pd.read_csv('./data/Social_Network_Ads.csv')

# 순서2 x,y추출  UserID  Gender  Age  EstimatedSalary  Purchased 
x = df.loc[ : , 'Age':'EstimatedSalary'] 
y = df['Purchased'] 


# # 순서3 전처리과정 스케일링 조정  MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler_x  = MinMaxScaler()
x = scaler_x.fit_transform(x)


# # 순서4 train_test_split(x,y,test_size) 확인  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1)    

 
# 순서5 분류에 관련 모델  구매여부(1구매, 0비구매) 
from sklearn.neighbors import  KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

# 순서6 fit(), y_pred = predict()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# 순서7 정확도 
from sklearn.metrics  import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)
# print(cm)  #지표값출력
print('수제 정확도', (50+39)/cm.sum())  # 0.89

from sklearn.metrics import  accuracy_score
ss = accuracy_score(y_test, y_pred) # 0.89
print('공식 정확도' , ss)
 

from sklearn.metrics import classification_report
print('KNeighborsClassifier  평가지표 보고서')
print(classification_report(y_test, y_pred))
print()

from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1,stop = x_set[:, 1].max() + 1, step = 0.01))

plt.figure(figsize=[10,7])
plt.contourf(X1, X2, model.predict(
            np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'blue'))(i), label = j)
    #~~ c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('KNeighborsClassifier 예측 결과 분류 (Test set)') 
plt.legend()
plt.show()
print()
print()








