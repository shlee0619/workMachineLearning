# 01ml.py

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

#난방비, 소득금액, 외식비, 가족인원, 외식횟수, 난방온도
#스케일 조정함수 StandardScaler, MinMaxScaler, MaxAbsSaler
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler

x = df.loc[:, 'Age':'Salary']
ss_x = StandardScaler()
print(ss_x.fit_transform(x))
print()

x = df.loc[:, 'Age':'Salary']
mm_x = MinMaxScaler()
print(mm_x.fit_transform(x))


