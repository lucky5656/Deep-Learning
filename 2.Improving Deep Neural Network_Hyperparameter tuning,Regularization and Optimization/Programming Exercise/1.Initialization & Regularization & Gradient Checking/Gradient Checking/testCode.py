import numpy as np
import gradient_checking


def gradient_check_n_test_code():
    np.random.seed(1)
    x = np.random.randn(4, 3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5, 4)
    b1 = np.random.randn(5, 1)
    W2 = np.random.randn(3, 5)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return x, y, parameters


# 测试forward_propagation
print("-----------------测试forward_propagation-----------------")
x, theta = 2, 4
J = gradient_checking.forward_propagation(x, theta)
print("J = " + str(J))

# 测试backward_propagation
print("-----------------测试backward_propagation-----------------")
x, theta = 2, 4
dtheta = gradient_checking.backward_propagation(x, theta)
print("dtheta = " + str(dtheta))

# 测试gradient_check
print("-----------------测试gradient_check-----------------")
x, theta = 2, 4
difference = gradient_checking.gradient_check(x, theta)
print("difference = " + str(difference))

X, Y, parameters = gradient_check_n_test_code()
cost, cache = gradient_checking.forward_propagation_n(X, Y, parameters)
gradients = gradient_checking.backward_propagation_n(X, Y, cache)
difference = gradient_checking.gradient_check_n(parameters, gradients, X, Y)
print("difference = " + str(difference))
