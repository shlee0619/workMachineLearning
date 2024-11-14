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
seed = np.random.RandomState(19937) 

x = 10 * seed.rand(100)
y = (x*2) - 1*seed.rand(100)
print()

x = x.reshape(100,1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
model = LinearRegression() #모델생성
model.fit(x,y)

print('모델학습 인자 나열 ', model.get_params())
print()
print('model.coef' , model.coef_ ) #[1.98406753] 정수값 
print('model.intercept' , model.intercept_ ) #-0.3855848856000055

#학습이론적용 데이터차원 2차원
#순수넘피 데이터일때 reshape( )
#모델의 인자확인 model.get_params()
#예측할때 predict(x_new),  (x_new*model.coef_)  + model.intercept_
x_new = np.linspace(-1,11,100)
x_new = x_new.reshape(-1,1)

y_predict = model.predict(x_new)
y_predict2 = (x_new*model.coef_)  + model.intercept_ 
print('- ' * 70)
print(y_predict)
print()
print(y_predict2)
print('- ' * 70)

plt.scatter(x,y)
plt.plot(x_new, y_predict, c='red')
plt.show()

print()
print()
# x = df.loc[ : , 'MedInc':'AveOccup']
# y = df['cost']

# from sklearn.model_selection import train_test_split
# x = df.loc[ : , 'MedInc':'AveOccup']
# y = df['cost']
# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, shuffle=False, random_state=0)

# 훈련데이터분리,테스트데이터분리
# 학습모델 LinearRegression, fit훈련, predict예측
# 예측한결과 시각화 표현 scatter스캐터
# 모델평가 정답률, 에러률
# 오차처리 최소 from sklearn.metrics import mean_squared_error, accuracy_score,root_mean_squared_error



# base = np.random.RandomState(0)
# source = np.random.RandomState(7)
# seed = np.random.RandomState(29937) #19937

# print('base값 =', base)     #0
# print('source값 =', source) #7
# print('seed값 =', seed)     #19937
# print()

# a = 10 * base.rand(10)
# b = 10 * source.rand(10)
# c = 10 * seed.rand(10)
# print(a); print()
# print(b); print()
# print(c); print()

# np.random.seed(123)
# d = 10 * seed.rand(10)
# print(d); print()
# print('- ' * 60)

# base = np.random.RandomState(19937)
# x = 10 * base.rand(5)
# print(x)
# print()

# np.random.seed(123)
# a = 10 * np.random.rand(5)
# b = 10 * np.random.rand(5)
# print(a)
# print(b)












print()
print()