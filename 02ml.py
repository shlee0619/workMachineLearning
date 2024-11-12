# 02ml.py

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
print(df)
 
print(': ' * 60)
print()

df = df.fillna(df.mean(numeric_only=True))
print(df) 

print()
print()
print('X,Y데이터분리 학습할 변수와 레이블링변수 분리')


# ML, DL학습에서 X대문자,Y대문자  본인각자 취향대로 대문자화, 소문자화 
# ML, DL학습에서 x소문자,y소문자
# scikit-learn에서 데이터는 대문자 X로 표기하고 레이블은 소문자 y로 표기
# 수학표기방식 2차원 배열행렬 원칙에서 대문자 X를 타깃은 1차원 배열의 벡터이므로 소문자 y를 사용


X = df.loc[ : , 'Country':'Salary' ]
y = df['Purchased']
print('X값데이터 ')
print(X)
print()
print('y값레이블링 ')
print(y)
print()

'''
                                         y값레이블링화함
    Country   Age      Salary            Purchased
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

# print() #df대신 X사용  X = df.loc[ : , 'Country':'Salary' ] 추출 
# print(X['Country'].unique())

print()
print( sorted(X['Country'].unique()) )


print()
print()