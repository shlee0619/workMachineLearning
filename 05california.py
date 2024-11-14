import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc ,  font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import matplotlib as mpl 
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False


import pandas as pd
import seaborn as sns
import numpy as np
#---------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.neighbors  import KNeighborsRegressor
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error 
from sklearn.datasets  import fetch_california_housing



housing = fetch_california_housing()
housing_df = pd.DataFrame(housing['data'] , columns=housing['feature_names'])
housing_df['Price'] = housing['target'] 
print(housing_df)

#정답 x = housing_df.loc[ : , 'MedInc':'AveOccup']
#정답 y = housing_df['Price'] 
 
# x = housing_df.drop(columns=['Latitude', 'Longitude',  'Price'], axis=1, inplace=False)    
x = housing_df.loc[ : , 'MedInc':'Longitude']
y = housing_df['Price']

housing_df.hist(bins=20, figsize=(12,8))
plt.show()
print()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=False)

lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)
print('predict함수', lrmodel.predict(x_test))       # [1.3097 1.63349368 1.18977681 ... 0.73958039 0.86926153 1.13112808]
print('score함수', lrmodel.score(x_train, y_train)) # 0.5305040120758866 
print('score함수', lrmodel.score(x_test, y_test)) # 0.5305040120758866 


'''
회귀알고리즘  LinearRegression, KNeighborsRegressor, SVC, DecisionTreeRegressior, RandomForestRegressor, 
평가방법 mean_squared_error, r2_score,  root_mean_squared_error
'''


from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error 
y_pred = lrmodel.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('mse평가 ' , mse)
print('sqrt적용 ' , rmse)
print()
print()
print('- ' * 60)

coef = pd.Series(data=np.round(lrmodel.coef_ ,2), index=x.columns)
print(coef.sort_values(ascending=False))
'''
AveBedrms     0.71
MedInc        0.45
HouseAge      0.01
Population   -0.00
AveOccup     -0.01
AveRooms     -0.12
Latitude     -0.42
Longitude    -0.43
dtype: float64
'''

print()
# https://seaborn.pydata.org
# sns.regplot
# AveBedrms 0.71, MedInc 0.45, HouseAge 0.01, Population -0.00
x_features = ['AveBedrms', 'MedInc', 'HouseAge','Population']
fig, axs = plt.subplots(figsize=(10, 8), ncols=2, nrows=2)
for i, feature in enumerate(x_features):
    row = int(i/2)
    col  = i%2
    sns.regplot(x=feature, y=housing_df['Price'], data=housing_df, ax=axs[row][col])

plt.show()




print()
print()