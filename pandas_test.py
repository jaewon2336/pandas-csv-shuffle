import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 도미
bream_length = pd.read_csv("test/bream_length.csv", header=None).to_numpy()
bream_weight = pd.read_csv("test/bream_weight.csv", header=None).to_numpy()

# 빙어
smelt_length = pd.read_csv("test/smelt_length.csv", header=None).to_numpy()
smelt_weight = pd.read_csv("test/smelt_weight.csv", header=None).to_numpy()

# 셔플 전 산점도
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

# 2차원 배열
bream_data = np.column_stack((bream_length, bream_weight))  # (35, 2)
smelt_data = np.column_stack((smelt_length, smelt_weight))  # (14, 2)

fish_data = np.concatenate((bream_data, smelt_data), axis=0)  # (49, 2)

target_data = np.array([1]*35 + [0]*14)  # 타겟 데이터

# shuffle
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

print(index)

# 훈련 데이터, 테스트 데이터 나누기
# numpy 리스트를 인덱스로 전달 가능
train_input = fish_data[index[:35]]
tarin_target = target_data[index[:35]]

test_input = fish_data[index[35:]]
test_target = target_data[index[35:]]

print(fish_data[13], train_input[0])

# 셔플 후 산점도
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel("length")
plt.ylabel("weight")
# plt.show()
