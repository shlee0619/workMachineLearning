from sklearn import svm, metrics
from sklearn.model_selection import KFold
import random, re

# 파일 읽기
try:
    with open('iris.csv', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
except FileNotFoundError:
    print("파일이 존재하지 않습니다. 'iris.csv' 파일을 확인하세요.")
    exit()

# 데이터 전처리
def to_numeric(value):
    return float(value) if re.match(r'^[0-9\.]+$', value) else value

data = [list(map(to_numeric, line.split(','))) for line in lines[1:]]
random.shuffle(data)  # 데이터 섞기

# 데이터와 라벨 분리
def split_data_label(rows):
    data, labels = [], []
    for row in rows:
        data.append(row[:4])
        labels.append(row[4])
    return data, labels

# K-Fold 교차 검증
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in kf.split(data):
    train_set = [data[i] for i in train_idx]
    test_set = [data[i] for i in test_idx]

    train_data, train_labels = split_data_label(train_set)
    test_data, test_labels = split_data_label(test_set)

    # SVM 학습 및 평가
    clf = svm.SVC()
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    accuracy = metrics.accuracy_score(test_labels, predictions)
    scores.append(accuracy)

# 결과 출력
print("각 Fold의 정확도:", scores)
print("평균 정확도:", sum(scores) / len(scores))
