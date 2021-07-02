import matplotlib.pyplot as plt
import reg_utils
import regularization


# matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset()  # 读取并绘制数据集
plt.show()


# 不使用正则化训练模型
parameters = regularization.model(train_X, train_Y, is_plot=True)
# 预测
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# 决策边界
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)


### L2正则化
# 使用L2正则化训练模型
parameters = regularization.model(train_X, train_Y, lambd=0.7,is_plot=True)
# 预测
print("使用正则化，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用正则化，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# 决策边界
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)


### dropout
# 使用dropout训练模型
parameters =regularization.model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=True)
# 预测
print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# 决策边界
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)
