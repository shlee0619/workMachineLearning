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
df = pd.DataFrame( housing['data'] , columns=housing['feature_names'])
print(df)
#문제   MedInc  HouseAge  ~~  Longitude  Cost필드/Price필드생활비정답

df['cost'] = housing.target
print(df)
#마지막열 생활비=정답지를 test데이터뽑고 

















print()
print()