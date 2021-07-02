import numpy as np
import matplotlib.pyplot as plt
import tf_utils
import time
import nn_TensorFlow

# 使用TensorFlow构建你的第一个神经网络:step1:创建计算图;step2:运行计算图。
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()  # 加载数据集
# X_train_orig.shape=(1080, 64, 64, 3); Y_train_orig.shape=(1, 1080);
# X_test_orig.shape=(120, 64, 64, 3); Y_train_orig.shape=(1, 120);
# classes=[0 1 2 3 4 5]

# 看一下数据集里面有什么，当然也可以自己更改一下index的值
index = 108
plt.imshow(X_train_orig[index])
print("Y = " + str(np.squeeze(Y_train_orig[:, index])))  # squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
plt.show()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # 每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# X_train_flatten.shape=(12288, 1080); X_test_flatten.shape=(12288, 120);

# 归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# 转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)
# X_train.shape=(12288, 1080);Y_train.shape=(6, 1080)
# X_test.shape=(12288, 120);Y_test.shape=(6, 120)


# 开始时间
start_time = time.perf_counter()
# 开始训练
parameters = nn_TensorFlow.model(X_train, Y_train, X_test, Y_test)
# 结束时间
end_time = time.perf_counter()
# 计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")
