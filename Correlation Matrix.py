import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv("./data/7.lpg_leakage.csv")

# 관심 있는 변수들만 선택
variables_of_interest = ['Alcohol', 'CH4', 'CO', 'H2', 'LPG', 'Propane', 'Smoke', 'Temp', 'LPG_Leakage']
data_selected = data[variables_of_interest]

# 상관계수 계산
correlation_matrix = data_selected.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()