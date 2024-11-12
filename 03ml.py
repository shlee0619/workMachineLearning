# 03ml.py

import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import numpy as np
import pandas as pd
#---------------------------------------------------------------------------------------------

df = pd.read_csv('./data/myData.csv')
df = df.fillna(df.mean(numeric_only=True))
print(df) 

print()
print()

#11-12-화요일
from sklearn.model_selection import train_test_split
x = df.loc[ : , 'Country':'Salary']
y = df['Purchased'] 

#train_test_split(x,y, test_size=0.2,shuffle=False, random_state=3)
x_train,x_test,y_train,y_test = train_test_split(x,y, shuffle=False)
print('x데이터 ')
print(x_train)
print()
print('y데이터 ')
print(y_train)

# 모델학습알고리즘 객체생성
# 훈련 fit()
# 정확도 predict()
# 예측


'''
   Country        Age        Salary Purchased
0   France  44.000000  72000.000000        No
1    Spain  27.000000  48000.000000       Yes
2  Germany  30.000000  54000.000000        No
3    Spain  38.000000  61000.000000        No
4  Germany  40.000000  63777.777778       Yes
5   France  35.000000  58000.000000       Yes
6    Spain  38.777778  52000.000000        No
7   France  48.000000  79000.000000       Yes
8  Germany  50.000000  83000.000000        No
9   France  37.000000  67000.000000       Yes
'''












print()
print()