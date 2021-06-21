# import library we need
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.transform import resize
from lr_utils import load_dataset


'### 加载数据集'
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


'### 数据预处理(一般步骤)：'
# 1、识别数据集的大小和形状
# 2、重塑一些数据的形状
# 3、“标准化”数据

# Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
m_train = train_set_x_orig.shape[0]  # m_train = 209
m_test = test_set_x_orig.shape[0]  # m_test = 50
num_px = train_set_x_orig[0].shape[0]  # Height/Width of each image: num_px = 64
# Each image is of size: (64, 64, 3)
# train_set_x shape: (209, 64, 64, 3) = (209, 12288)
# train_set_y shape: (1, 209)
# test_set_x shape: (50, 64, 64, 3) = (50, 12288)
# test_set_y shape: (1, 50)

# Reshape the training and test examples
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T  # train_set_x_flatten shape:(12288, 209)
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T  # test_set_x_flatten shape: (12288, 50)

# standardize our dataset（Image normalization）
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


'### 算法的创建基本流程是:'
#  1、定义模型的架构(比如输入特征的数量)
#  2、初始化模型参数
#  3、循环
#     1)前向传播计算当前的代价函数
#     2)反向传播计算当前的梯度
#     3)使用梯度下降的方法更新参数
#   4、分别实现1-3三个步骤，并且最终放在一个函数model ()中

# GRADED FUNCTION: sigmoid
# Step1：实现sigmod函数
def sigmoid(z):
    """
    参数：
        z  - 任何大小的标量或numpy数组。
    
    返回：
        s  -  sigmoid（z）
    """
    s = 1 / (1 + np.exp(-z))
    return s


# GRADED FUNCTION: initialize_with_zeros
# Step2：初始化参数w、b
def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。
        
        参数：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）
        
        返回：
            w  - 维度为（dim，1）的初始化向量。
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros((dim, 1))  # w的维度是(dim,1)，dim = (num_px *num_px *3, 1)
    b = 0
    assert (w.shape == (dim, 1))   # assert（）断言函数是为了检测一下是否正常，来确保我要的数据是正确的
    assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或者是int
    return w, b


# GRADED FUNCTION: propagate
# Step3：前向传播与反向传播
def propagate(w, b, X, Y):
    """
    实现前向和后向传播的成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w相同的形状
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    m = X.shape[1]
    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))
    # 反向传播
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    # 使用断言确保数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)  # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
    assert (cost.shape == ())
    # 创建一个字典，把dw和db保存起来
    grads = {"dw": dw, "db": db}
    return grads, cost


# GRADED FUNCTION: optimize
# Step4：参数更新（优化）
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
     """
    此函数通过运行梯度下降算法来优化w和b
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值
    
    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。
    
    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """
     costs = []
     for i in range(num_iterations):
         grades, cost = propagate(w, b, X, Y)
         dw = grades["dw"]
         db = grades["db"]
         w = w - learning_rate * dw
         b = b - learning_rate * db

         if i % 100 == 0:  # 每迭代100次更新一次参数
             costs.append(cost)
         if print_cost and i % 100 == 0:
             print("Cost after iteration %i: %f" % (i, cost))

     params = {"w": w, "b": b}

     grads = {"dw": dw, "db": db}

     return params, grads, costs


# GRADED FUNCTION: predict
# Step5：利用训练好的模型对测试集进行预测
def predict(w, b, X):
     """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据
    
    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
    
    """
     m = X.shape[1]
     Y_prediction = np.zeros((1, m))
     w = w.reshape(X.shape[0], 1)
     A = sigmoid(np.dot(w.T, X) + b)
     for i in range(A.shape[1]):
         if (A[0, i] > 0.5):
             Y_prediction[0, i] = 1
         else:
             Y_prediction[0, i] = 0

     assert (Y_prediction.shape == (1, m))
     return Y_prediction


# GRADED FUNCTION: model
# Step6：将以上功能整合到一个模型中
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    
    返回：
        d  - 包含有关模型信息的字典。
    """
    w, b = np.zeros((X_train.shape[0], 1)), 0  # 初始化参数w、b
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)  # 优化参数
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)  # 预测
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

'### 把训练好的模型参数储存起来'
# 先创建并打开一个文本文件
file = open('模型参数.txt', 'w')
# 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
for k, v in d.items():
    file.write(str(k) + ':' + str(v) + '\n')
# 注意关闭文件
file.close()

'### 测试集中具体的图片预测情况'
index = 40
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
    int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")
plt.show()

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# # 学习率优化 learning_rates = [0.01, 0.001, 0.0001] models = {} for i in learning_rates: print("learning rate is: " +
# str(i)) models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500,
# learning_rate=i, print_cost=False)
# print('\n' + "-------------------------------------------------------" + '\n')
#
# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))  # 设置图例(legend)的标
#
# plt.ylabel('cost')
# plt.xlabel('iterations')
#
# legend = plt.legend(loc='upper center', shadow=True)  # legend显示图例 # shadow设置背景为灰色
# frame = legend.get_frame()  # get_frame() 返回legend所在的方形对象 # 获得背景
# frame.set_facecolor('0.90')  # 设置图例legend背景透明度
# plt.show()


# 测试自己的图片
my_image = "16.jpg"  # change this to the name of your image file

# We preprocess the image to fit your algorithm.
fname = "C:/Users/123/Pictures/" + my_image
image = np.array(imageio.imread(fname))

my_image = resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
# my_image = my_image / 255.
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \""
      + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
plt.show()
