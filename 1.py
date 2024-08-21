import pandas as pd
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
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
# print(model.coef_, model.intercept_)

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