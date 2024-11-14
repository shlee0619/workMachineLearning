import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import matplotlib as mpl # 음수 표시 에러 
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
#---------------------------------------------------------------------------------------------

df = pd.read_csv('./data/myData.csv')
#결측값전처리 df = df.fillna(df.mean(numeric_only=True))
#결측값전처리 print(df) 
print(df)
print()


x = df.loc[ : , 'Country' : 'Salary' ]
# sorted( x['Country'].unique() ) #['France', 'Germany', 'Spain']


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = LabelEncoder()
x['Country'] = encoder.fit_transform( x['Country'] )
print(x)
print('국가표시를 숫자화 F1 S3 G2 LabelEncoder() encoder.fit_transform()')
print()


# x = df.loc[ :  , 'Country' : 'Salary' ]
x = df[['Age', 'Salary', 'Country']]
print('- ' * 50)


np.set_printoptions(suppress=True)   #remainder='passthrough'
#생략하면 넘피 array형태  ct = ColumnTransformer([('encoder', OneHotEncoder(), [2])] )
#에러 ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ct = ColumnTransformer([('test', OneHotEncoder(), [2])], remainder='passthrough')
x = ct.fit_transform(x)
myone = np.asarray(x, dtype=int)
print(myone)
#실수형출력 비권장 print(x) 
'''
[[    1     0     0    44 72000]
 [    0     0     1    27 48000]
 [    0     1     0    30 54000]
 [    0     0     1    38 61000]
 [    0     1     0    40 63777]
 [    1     0     0    35 58000]
 [    0     0     1    38 52000]
 [    1     0     0    48 79000]
 [    0     1     0    50 83000]
 [    1     0     0    37 67000]]
'''
print()
print()