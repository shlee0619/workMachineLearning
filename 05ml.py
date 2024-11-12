import time
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

#난방비, 소득금액, 외식비, 가족인원, 외식횟수, 난방온도 
#스케일 조정함수 StandardScaler같이,MinMaxScaler같이,MaxAbsScaler각자
# StandardScaler 평균을기준으로 평균에서 얼마만큼 떨어져 있느냐?
# MinMaxScaler 0~1사이 숫자 맞춤  데이터 최소최대값 알고 있을때 사용

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler

x = df.loc[ : , 'Age':'Salary' ]  #x = df.loc[ : , 'Age':'Salary' ]
ss_x = StandardScaler()
print(ss_x.fit_transform(x))
print()

x = df.loc[ : , 'Age':'Salary' ]
mm_x = MinMaxScaler()
print(mm_x.fit_transform(x))

# ma_x = MaxAbsScaler() 


# 정시 5시10분
# 재시험 4시 10분

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