import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
import tf_utils
import nn_TensorFlow


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()  # 加载数据集
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # 每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255
# 转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig, 6)
parameters = nn_TensorFlow.model(X_train, Y_train, X_test, Y_test)

for i in range(6):
    my_image = str(i) + ".png"  # 定义图片名称
    fileName = "image1/" + my_image  # 图片地址
    image = mpimg.imread(fileName)  # 读取图片
    plt.imshow(image)  # 显示图片
    plt.show()

    my_image = image.reshape(1, 64 * 64 * 3).T  # 重构图片
    my_image_prediction = tf_utils.predict(my_image, parameters)  # 开始预测
    print("预测结果: y = " + str(np.squeeze(my_image_prediction)), "真实结果：y=" + str(i))

for j in range(6):
    my_image = str(j) + ".png"  # 定义图片名称
    fileName = "image2/" + my_image  # 图片地址
    image = mpimg.imread(fileName)  # 读取图片
    plt.imshow(image)  # 显示图片
    plt.show()

    my_image = image.reshape(1, 64 * 64 * 3).T  # 重构图片
    my_image_prediction = tf_utils.predict(my_image, parameters)  # 开始预测
    print("预测结果: y = " + str(np.squeeze(my_image_prediction)), "真实结果：y=" + str(j))
