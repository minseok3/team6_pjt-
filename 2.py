import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection  import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
import seaborn as sns

data = pd.read_csv("./data/7.lpg_leakage.csv")
print(data)
# 데이터 전처리 : Min-Max 전처리
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

# 모델 선택 및 분할
model = DecisionTreeClassifier()
# model = LogisticRegression()
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
confu = confusion_matrix(Y_pred, Y_test)
print(confu)

# model = LogisticRegression()
model.fit(X_train, Y_train)

# 예측값 생성
y_pred = model.predict(X_test)

fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring="accuracy")
print(acc)

sum_acc = sum(acc)/len(acc)
print(sum_acc)

# fold = KFold(n_splits=10, shuffle=True)
# mse = cross_val_score(model, X_train, Y_train, cv=fold, scoring="neg_mean_squared_error")
# print(mse.mean())

# MSE = mean_squared_error(Y_test, Y_pred)
# print(MSE)

# 결과(모델 예측값 vs 실제값) 시각화
plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test[:300])), Y_test[:300], color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(Y_pred[:300])), Y_pred[:300], color='red', label='Predicted Values', marker='x')

plt.title("Actual-Predicted")
plt.xlabel("Environment")
plt.ylabel("LPG-Leakage")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')

# confusion matrix
cf_matrix = confusion_matrix(Y_pred, Y_test)

# annotation 준비
group_names = ["TN", "FP (type 1 error)", "FN (type 2 error)", "TP"]

# T(True) : 예측값과 실제값이 같은가 / F(False) : 예측값과 실제값이 다른가
# Positive : 긍정(1) / Negative : 부정(0)
# True Positive(TP) : 실제 True인 정답을 True라고 예측 (정답)
# True Negative(TN) : 실제 False인 정답을 False라고 예측 (정답)
# False Positive(FP) : 실제 False인 정답을 True라고 예측 (오답)
# False Negative(FN) : 실제 True인 정답을 False라고 예측 (오답)

group_counts = [value for value in cf_matrix.flatten()]
group_percentages = [f"{value:.1%}" for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

# 시각화
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')
