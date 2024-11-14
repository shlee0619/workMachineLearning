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
print(df)
print()

#결측값전처리 df = df.fillna(df.mean(numeric_only=True))
#결측값전처리 print(df) 

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
print()






print()
print()