import numpy as np
from Build_DNN_Step_by_Step import *

# 初始化参数
# 两层
print("initialize_parameters()测试结果：")
parameters = initialize_parameters(3, 2, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
# L层
print("initialize_parameters_deep()测试结果：")
parameters = initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 两层前向传播
# 线性传播部分
def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A, W, b


print("linear_forward()测试结果：")
A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))


# 激活部分
def linear_activation_forward_test_case():
    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A_prev, W, b


print("linear_activation_forward()测试结果：")
A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
print("With ReLU: A = " + str(A))


# L层前向传播
def L_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters


print("L_model_forward()测试结果：")
X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# 代价函数
def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8, .9, 0.4]])

    return Y, aL


print("compute_cost()测试结果：")
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))


# 两层反向传播
# 线性传播部分
def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    linear_cache = (A, W, b)
    return dZ, linear_cache


print("linear_backward()测试结果：")
# Set up some test inputs
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print("dA_prev = " + str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))


# 激活部分
def linear_activation_backward_test_case():
    """
    aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
    """
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache


print("linear_activation_backward()测试结果：")
AL, linear_activation_cache = linear_activation_backward_test_case()

print("linear_activation_backward(activation=sigmoid)测试结果：")
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
print("sigmoid:")
print("dA_prev = " + str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db) + "\n")

print("linear_activation_backward(activation=relu)测试结果：")
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu")
print("relu:")
print("dA_prev = " + str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))


# L层反向传播
def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches


print("L_model_backward()测试结果：")
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dA1 = " + str(grads["dA1"]))


# 更新参数
def update_parameters_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dW2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads


print("update_parameters()测试结果：")
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))