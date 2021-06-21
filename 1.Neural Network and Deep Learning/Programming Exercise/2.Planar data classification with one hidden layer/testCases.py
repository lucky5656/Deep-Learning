"""
# WANGZHE12
"""
import numpy as np
from NN_model import layer_sizes,initialize_parameters,forward_propagation,compute_cost,backward_propagation,update_parameters,nn_model


# 定义神经网络的结构
def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess
# 测试
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


# 初始化模型参数
def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y
# 测试
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 循环：1、前向传播
def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}
    return X_assess, parameters
# 测试
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
# Note: we use the mean here just to make sure that your output matches ours.
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))


# 循环：2、交叉熵J计算（计算损失,代价函数）
def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}
    a2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))
    return a2, Y_assess, parameters
# 测试
A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess)))


# 循环：3、执行反向传播
def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}
    cache = {'A1': np.array([[-0.00616578, 0.0020626, 0.00349619],
                             [-0.05225116, 0.02725659, -0.02646251],
                             [-0.02009721, 0.0036869, 0.02883756],
                             [0.02152675, -0.01385234, 0.02599885]]),
             'A2': np.array([[0.5002307, 0.49985831, 0.50023963]]),
             'Z1': np.array([[-0.00616586, 0.0020626, 0.0034962],
                             [-0.05229879, 0.02726335, -0.02646869],
                             [-0.02009991, 0.00368692, 0.02884556],
                             [0.02153007, -0.01385322, 0.02600471]]),
             'Z2': np.array([[0.00092281, -0.00056678, 0.00095853]])}
    return parameters, cache, X_assess, Y_assess
# 测试
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print("dW1 = "+ str(grads["dW1"]))
print("db1 = "+ str(grads["db1"]))
print("dW2 = "+ str(grads["dW2"]))
print("db2 = "+ str(grads["db2"]))


# 循环：4、更新参数
def update_parameters_test_case():
    parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                                  [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07],
                                  [8.15562092e-06],
                                  [6.04810633e-07],
                                  [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}
    grads = {'dW1': np.array([[0.00023322, -0.00205423],
                              [0.00082222, -0.00700776],
                              [-0.00031831, 0.0028636],
                              [-0.00092857, 0.00809933]]),
             'dW2': np.array([[-1.75740039e-05, 3.70231337e-03, -1.25683095e-03,
                               -2.55715317e-03]]),
             'db1': np.array([[1.05570087e-07],
                              [-3.81814487e-06],
                              [-1.90155145e-07],
                              [5.46467802e-07]]),
             'db2': np.array([[-1.08923140e-05]])}
    return parameters, grads
# 测试
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 模型融合，组合成nn_model()
def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess
# 测试
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 预测
def predict_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                                  [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07],
                                  [8.15562092e-06],
                                  [6.04810633e-07],
                                  [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}
    return parameters, X_assess
# import planar_data_classification as predice 报错：cannot import name 'predict' from 'planar_data_classification'
#就把predict函数复制过来了
def predict(parameters, X):
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    # predictions = np.around(A2)#这种方式也可以
    print("predictions:", predictions)
    return predictions
# 测试
parameters, X_assess = predict_test_case()
predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))
