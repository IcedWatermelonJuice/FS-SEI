import numpy as np

# 读取16类数据集
X16_train = np.load("WiFi_ft62/X_train_16Class.npy")
Y16_train = np.load("WiFi_ft62/Y_train_16Class.npy")

X16_test = np.load("WiFi_ft62/X_test_16Class.npy")
Y16_test = np.load("WiFi_ft62/Y_test_16Class.npy")

# 切割为10类数据集
mask10_train = Y16_train < 10
X10_train = X16_train[mask10_train]
Y10_train = Y16_train[mask10_train]

mask10_test = Y16_test < 10
X10_test = X16_test[mask10_test]
Y10_test = Y16_test[mask10_test]

# 切割为6类数据集
mask6_train = Y16_train >= 10
X6_train = X16_train[mask6_train]
Y6_train = Y16_train[mask6_train]
Y6_train = Y6_train - 10

mask6_test = Y16_test >= 10
X6_test = X16_test[mask6_test]
Y6_test = Y16_test[mask6_test]
Y6_test = Y6_test - 10

# 保存切割后的数据集
np.save("WiFi_ft62/X_train_10Class", X10_train)
np.save("WiFi_ft62/Y_train_10Class", Y10_train)
np.save("WiFi_ft62/X_test_10Class", X10_test)
np.save("WiFi_ft62/Y_test_10Class", Y10_test)
np.save("WiFi_ft62/X_train_6Class", X6_train)
np.save("WiFi_ft62/Y_train_6Class", Y6_train)
np.save("WiFi_ft62/X_test_6Class", X6_test)
np.save("WiFi_ft62/Y_test_6Class", Y6_test)
