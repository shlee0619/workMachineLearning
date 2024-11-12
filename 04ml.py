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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

print('0neHotEncoder처리 10:39')
x = df.loc[ : , 'Country':'Salary']
y = df['Purchased']


encoder = LabelEncoder()
