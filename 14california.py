import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)


import pandas as pd
import seaborn as sns
import numpy as np
#---------------------------------------------------------------------------------------------
x = np.array([6, 8, 10, 15, 18])
y = np.array([7, 9, 13, 17.5, 19.3])


plt.plot(x, y, 'ro')  # ro=red+dot의미  v역삼각형, ^삼각형  s=스케어 +플러스기호 
plt.xlabel('제품크기 inches')
plt.ylabel('prices')
plt.axis([0, 25, 0, 25])
plt.grid()
plt.show()


def my_linear1(data):
    return data * 1.3 - 0.1
    

def my_linear2(data):
    return data * 0.9 - 1.05

plt.plot(x, y, 'ro')
plt.plot(x, my_linear1(x), 'k--')   # ro=red+dot의미  k=블랙  -라인  w=white
plt.plot(x, my_linear2(x), 'g--')  
plt.xlabel('제품크기 inches')
plt.ylabel('prices')
plt.axis([0, 25, 0, 25])
plt.grid()
plt.show()


from sklearn.linear_model import LinearRegression # 임포트 해서 
model=LinearRegression()   
print(model.get_params())
'''
copy_X= : 입력 데이터의 복사 여부
fit_intercept= : 절편의 값을 계산 여부
normalize= : 정규화여부
n_jobs= : 데이터 분석에 사용할 코어의 갯수(기본값은 1인데 -1을 입력하는 경우 사용가능한 모든 코어를 사용)
'''

x = x.reshape(-1,1)
print(x) #([6] [8], [10], [15], [18]])
model.fit(x,y)
print()


# x = np.array([6, 8, 10, 15, 18])
# y = np.array([7, 9, 13, 17.5, 19.3])
pred = model.predict(x)
print('학습예측 : ' , pred) #학습예측:[ 7.51068548  9.60302419 11.6953629  16.92620968 20.06471774]
print()
score = model.score(x,y)
print('학습평가 : ' , score) #학습평가 :  0.9710179197685516
print()

plt.plot(x, y, 'ro')
plt.plot(x, model.predict(x), 'k--')
plt.xlabel('제품크기 inches')
plt.ylabel('prices')
plt.axis([0, 25, 0, 25])
plt.grid()
plt.show()

# LinearRegression선형회귀공식 y = ax+b 공식을 자동으로 계산해줌
print('기울기=', model.coef_, '바이어스=' , model.intercept_)
print((x*model.coef_) + model.intercept_)
print()

predict_1 = model.predict(x)
print(predict_1)
print()
print()


print('- ' * 60)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
df = pd.DataFrame( housing['data'] , columns=housing['feature_names'])
print(df)
print()

x = pd.DataFrame( housing['data'] , columns=housing['feature_names'])
print(x)
print()

y = pd.Series( housing['target'] )
print(y)
print()

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=1)
print()
print()


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
k=4

knn_model=KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train)
lr_model=LinearRegression().fit(x_train, y_train)
print('학습 평가(KNN) :',  knn_model.score(x_train, y_train)) #0.5006155384647328
print('학습 평가(LR) :',   lr_model.score(x_train, y_train))  #0.6102859678113063
print()
print()

print('테스트 평가(KNN) :',  knn_model.score(x_test, y_test)) # 0.12211572715320917
print('테스트 평가(LR) :',   lr_model.score(x_test, y_test))  # 0.5929869285760037
print()
print()
print()










print()
print()





