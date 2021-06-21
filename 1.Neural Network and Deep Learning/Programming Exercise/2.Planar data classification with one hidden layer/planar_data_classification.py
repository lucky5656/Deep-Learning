# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from NN_model import layer_sizes,initialize_parameters,forward_propagation,compute_cost,backward_propagation,update_parameters,nn_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# matplotlib inline
np.random.seed(1)  # set a seed so that the results are consistent


# 加载数据
X, Y = load_planar_dataset()
# 可视化数据
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.show()
# 获取数据的维度
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]  # training set size
# 测试
print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))


# 简单的逻辑回归
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 预测X, Y对应坐标,绘制预测后的决策边缘
plt.title("Logistic Regression")
# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")  # dot(Y,LR_predictions):1预测为1的个数；dot(1-Y,1-LR_predictions):0预测为0的个数
plt.show()


# 神经网络：NN_model()
# 初始化模型参数
# 循环：1、前向传播
# 循环：2、交叉熵J计算（计算损失,代价函数）
# 循环：3、执行反向传播
# 循环：4、更新参数
# 模型融合，组合成nn_model()
# 预测
# GRADED FUNCTION: predict
def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)  # A2矩阵中大于0.5的元素会被转为True,否则转为False
    # predictions = np.around(A2)#这种方式也可以
    # print("predictions:", predictions)
    return predictions


# 边界绘制
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()


# 准确率
# Print accuracy
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')


# 隐藏层神经元数量的影响（调整隐藏层大小）
# This may take about 2 minutes to run
# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(4, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations = 5000)
#     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
# plt.show()


# 其他数据集上的表现
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_circles"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
# make blobs binary
if dataset == "noisy_moons":
    Y = Y % 2
# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.show()


# 隐藏层神经元数量的影响（调整隐藏层大小）
# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(4, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations = 5000)
#     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
# plt.show()
