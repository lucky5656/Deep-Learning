import matplotlib.pyplot as plt
import init_utils   # 第一部分，初始化
import initialization

# matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = init_utils.load_dataset()
plt.show()


### 零初始化
# 使用零初始化的参数来训练模型
parameters = initialization.model(train_X, train_Y, initialization="zeros", is_polt=True)
# 预测
print("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))
# 预测和决策边界的细节
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)


### 随机初始化
# 使用随机初始化的参数来训练模型
parameters = initialization.model(train_X, train_Y, initialization="random", is_polt=True)
# 预测
print("训练集：")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集：")
predictions_test = init_utils.predict(test_X, test_Y, parameters)
print(predictions_train)
print(predictions_test)
# 分类的结果
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)


### 抑梯度异常初始化
# 使用抑梯度异常初始化的参数来训练模型
parameters = initialization.model(train_X, train_Y, initialization="he", is_polt=True)
# 预测
print("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
