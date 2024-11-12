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

housing = fetch_california_housing(as_frame=True)
print(housing.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
print('- ' * 60)
print('housing.data')
print(housing.data)
print()

print('housing.frame')
print(housing.frame) #[20640 rows x 9 columns]
print('11-12-화요일 2:4 ')


# californai데이터셋 데이터필드속성
# MedInc(중위 소득), Housing Age(주택 연식), 
# AveRooms(평균 방 개수), 
# AveBedrms(평균 침실 수), Population(인구 수), 
# AveOccup(평균 거주자 수), 
# Latitude(위도), Longitude(경도), MedHouseVal






print()
print()