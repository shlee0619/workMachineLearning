import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import matplotlib as mpl # 음수 표시
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False

import seaborn as sns
import numpy as np
import pandas as pd
#---------------------------------------------------------------------------------------------

# df = pd.read_csv('./data/myData.csv')
# df = df.fillna(df.mean(numeric_only=True))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()
df = pd.DataFrame( housing['data'] , columns=housing['feature_names'])
print(df)
print()

print()
print('필드의 결측값 갯수 출력')
print(df.isna().sum())
print()

#결측값을 그래프 작성  pip install missingno
import missingno as msno 
msno.matrix(df, figsize=(12,10))
plt.show()


df = df.fillna(df.mean(numeric_only=True))
print(df) 

df['cost'] = housing.target  #맨마지막열에 cost금액추가
print(df)
print()

df.hist(bins=20, figsize=(12,10))
df.hist(bins=25, figsize=(10,8))
plt.show()
print('- ' * 70)


# #sns의 histplot()사용
# sns.histplot(data = df, x="MedInc")
# plt.title('sns.histplot MedInc필드평균수입')
# plt.show()


# #sns.boxplot()사용
# sns.boxplot(data = df.loc[:, ["AveRooms", "AveBedrms", "AveOccup"]])
# plt.title('boxplot 수요일 ')
# plt.show()


# 판다스 데이터프레임 장점 
# df.hist( )차트접근
# df.corr()상관함수   

# plt.figure(figsize = (10, 6))
# #fmt단위확인  sns.heatmap(df.corr(),  annot=True, cmap='coolwarm', fmt='.2f', linewidths=1.5) 
# #비권장기본 sns.heatmap(df.corr(),  annot=True) 
# sns.heatmap(df.corr(),  annot=True, cmap='coolwarm',  linewidths=1.0) 
# plt.title('상관관계 heatmap')
# plt.show()
# print()



from sklearn.model_selection import train_test_split
x = df.loc[ : , 'MedInc':'AveOccup'] #문제   훈련문제,test문제
y = df['cost'] #정답  훈련정답, test정답

print(x)
print()
print(y)

# x데이터축  Latitude  Longitude   cost 제외
# y데이터축  cost 
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, shuffle=False, random_state=0)

print('x데이터 ')
print(x_train)
print()
print('y데이터 ')
print(y_train)
print()


from sklearn.linear_model import LinearRegression
lr = LinearRegression()  #모델생성 
lr.fit(x_train,y_train)  #데이터갯수만큼 돌려야 하는데 for,while대신 fit()

print('11-12-수요일 LinearRegression(), fit(1,2)')
predict = lr.predict(x_test)
print('예측결과비율 표시  ', predict )

















print()
print()