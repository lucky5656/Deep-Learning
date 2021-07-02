import matplotlib.pyplot as plt
import opt_utils
import optimization


train_X, train_Y = opt_utils.load_dataset(is_plot=True)
plt.title("Visualizing data sets")
plt.show()


# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]


# 使用普通的梯度下降
parameters1 = optimization.model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)
# 预测
preditions = opt_utils.predict(train_X, train_Y, parameters1)
# 绘制分类图
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters1, x.T), train_X, train_Y)


# 使用动量的梯度下降
parameters2 = optimization.model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum", is_plot=True)
# 预测
preditions = opt_utils.predict(train_X, train_Y, parameters2)
# 绘制分类图
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters2, x.T), train_X, train_Y)


# 使用Adam优化的梯度下降
parameters3 = optimization.model(train_X, train_Y, layers_dims, optimizer="adam", is_plot=True)
# 预测
preditions = opt_utils.predict(train_X, train_Y, parameters3)
# 绘制分类图
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters3, x.T), train_X, train_Y)

