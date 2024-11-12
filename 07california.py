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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing.keys())  #딕트구조 키값 
#['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR']
data = pd.DataFrame(housing['data'], columns=housing['feature_names'])
target = pd.DataFrame(housing['target'] , columns=['Target'])
df = pd.concat([data,target], axis=1) #기존꺼끝에 target맨마지막열 지정
print(df) #[20640 rows x 9 columns]

print()
print( '결측값확인  ' , df.isnull().sum())
print()

from sklearn.model_selection import train_test_split
# x = df.loc[ : , 'Country':'Salary']
# y = df['Purchased'] 



'''
housing.frame
       MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
0      8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526
1      8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585
2      7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24        3.521
       ...       ...       ...       ...        ...         ...       ...       ...        ...          ...
20638  1.8672      18.0  5.329513   1.171920       741.0  2.123209     39.43    -121.32        0.847
20639  2.3886      16.0  5.254717   1.162264      1387.0  2.616981     39.37    -121.24        0.894
'''
print(' test  test  2:10  2:11')
print('- ' * 60)
# californai데이터셋 데이터필드속성
# MedInc(중위 소득), Housing Age(주택 연식), 
# AveRooms(평균 방 개수), 
# AveBedrms(평균 침실 수), Population(인구 수), 
# AveOccup(평균 거주자 수), 
# Latitude(위도), Longitude(경도), MedHouseVal









print()
print()