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

a = np.linspace(-3,3,10)
print('np.linspace(-3,3,10) 결과')
print(a)
print()

b = np.linspace(11,20,10)
print('np.linspace(11,20,10) 결과')
print(b)
print()

c = np.arange(0,7)
print('np.arange(0,7) 결과')
print(c)
print()

d = np.arange(11,20)
print('np.arange(11,20) 결과')
print(d)
print()


e = np.arange(0.1, 1, 0.1, dtype=np.float64)
print('np.arange(0.1, 1, 0.1, dtype=np.float64) 결과')
print(e)
print()
print()


x = np.linspace(1,10,10)
y = np.linspace(11,20,10)
m,n = np.meshgrid(x,y)
plt.scatter(m,n)
plt.grid()
plt.show()











