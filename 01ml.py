# 01ml.py

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

import numpy as np
import pandas as pd
#---------------------------------------------------------------------------------------------

df = pd.read_csv('./data/myData.csv')
print(df)
print()
'''
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
'''

print('- ' * 55)
print()

print(df.describe()) #갯수, 최대,최소,편차, 1사분기
print()

print(df.info()) #필드정보
'''
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   Country    10 non-null     object
 1   Age        9 non-null      float64
 2   Salary     9 non-null      float64
 3   Purchased  10 non-null     object
dtypes: float64(2), object(2)
'''
print()
print(df['Country'].describe() )
# print(df.Country.describe() )
print( )

# print(df['Purchased'].describe() )
print(df.Purchased.describe() )
print( )

#문제1 Purchased  unique() 중복데이터 대표한건출력
print(df['Purchased'].unique() )
print( )

print(df['Country'].unique() )
print( )
print()
print('🎄 ' * 30 )

#문제2 결측값 데이터 필드별 결측값 NAN 몇건 
print('필드의 결측값 갯수 출력')
print(df.isna().sum())
'''
필드의 결측값 갯수 출력
Country      0
Age          1
Salary       1
Purchased    0
dtype: int64
'''
print()


#문제3 결측필드 데이터값 해결 - drop, 평균값대체, 중앙값대체 
# print()
# print()
# df.dropna(inplace=True)
# print(df)

'''
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4번  drop          NaN   
5   France  35.0  58000.0       Yes
6번 drop     NaN
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
'''


#문제4 결측필드 데이터값  평균값 대체 
#각자 해결  df.mean()함수이용 
print()
print(': ' * 60)
df = df.fillna(df.mean(numeric_only=True))
print(df) #df.head()

'''
   Country   Age   Salary Purchased
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