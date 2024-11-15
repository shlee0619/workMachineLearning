import time
import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)
import matplotlib as mpl

mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 순서1 Social_Network_Ads.csv 파일 읽기
path = './data/Social_Network_Ads.csv'
data = pd.read_csv(path)

# 순서2 x, y 추출
# 'Age'와 'EstimatedSalary'를 특징 변수로 설정, 'Purchased'를 목표 변수로 설정
x = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# 순서3 전처리과정 스케일링 조정 MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# 순서4 train_test_split(x, y, test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# 데이터가 정상적으로 분리되었는지 확인하는 출력
print("Train set X shape:", x_train.shape)
print("Test set X shape:", x_test.shape)
print("Train set Y shape:", y_train.shape)
print("Test set Y shape:", y_test.shape)

# Logistic Regression 모델 학습
model = LogisticRegression()
model.fit(x_train, y_train)

# 예측 수행
y_pred = model.predict(x_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# 시각화 (confusion matrix heatmap)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
