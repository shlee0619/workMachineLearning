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

from sklearn.preprocessing import LabelEncoder

x = df.loc[ : , 'Country':'Salary' ]
y = df['Purchased']

encoder = LabelEncoder()
x['Country'] = encoder.fit_transform(x['Country'])
print(x)
print()


print('- ' * 50)
print('y원래값 ')
print(y)
print()

encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)
print('yLabelEncoder() 결과값 ')
print(y)
print()

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
OneHotEncoder처리 10:39

'''






print()
print()