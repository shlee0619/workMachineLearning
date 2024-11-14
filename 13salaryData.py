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

df = pd.read_csv('./data/Salary_Data.csv')
print(df)
print()
print(df.info())
print()

#순서1단계 여러전처리 결측값 
print('전처리 결측값 ') #결측값
print(df.isna().sum()) 
print()


#순서2단계 x,y
x=df.iloc[:,0].values
print(x) #근무year
print()
y=df.iloc[:,1].values
print(y) #급여연봉
print()


#데이터분리  loc[] , iloc[] , train_test_split(x,y, test_size=0.2)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=False)
print('x_train')
print(x_train)
print()
print('y_train')
print(y_train)
#훈련데이터  23번까지  8.2  113812.0

print()
print()
print('x_train.shape', x_train.shape ) #(24,)
x_train = x_train.reshape(24,1)
print('x_train.shape ', x_train.shape ) #(24, 1)
print()

print('train전 학습')
plt.scatter(x, y, color='red')
plt.show()


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train) #훈련데이터 fit()

print('x_test.shape[0] 결과 ', x_test.shape[0]) #6개출력
x_test = x_test.reshape(6, 1) #수작업으로 6기술, 6대신 x_test.shape[0]
print()

np.set_printoptions(suppress=True)
predict = model.predict(x_test)
print('predict예측 ' , predict)
print()

predict2 = (x_test*model.coef_) + model.intercept_ 
print('predict예측2 ' , predict2)
print()

my1 = np.asarray(predict, dtype=int)
my2 = np.asarray(predict2, dtype=int)
print('my1')
print(my1)
print()
print('my2')
print(my2)
print('- ' * 60)

print()
print()


plt.scatter(x_train, y_train, color = 'red')
# plt.plot(x_test, y_test, color ='blue')
plt.show()
print()
print()



