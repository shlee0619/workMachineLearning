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

#11-12-화요일
from sklearn.model_selection import train_test_split

x = df.loc[:, 'Country':'Salary']
y = df['Purchased']

x = 1
y = 2
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=False, random_state=3)
print('x_train 데이터: ')
print(x_train)
print()
print('y_train데이터')
print(y_train)
