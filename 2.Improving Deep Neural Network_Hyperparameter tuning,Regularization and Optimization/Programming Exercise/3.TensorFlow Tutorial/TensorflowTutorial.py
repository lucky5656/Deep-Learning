import numpy as np
import tensorflow as tf

# %matplotlib inline #如果你使用的是jupyter notebook取消注释
np.random.seed(1)
y_hat = tf.constant(36, name="y_hat")  # 定义y_hat为固定值36
y = tf.constant(39, name="y")  # 定义y为固定值39
loss = tf.Variable((y - y_hat) ** 2, name="loss")  # 为损失函数创建一个变量
init = tf.compat.v1.global_variables_initializer()  # 运行之后的初始化(session.run(init))
# 损失变量将被初始化并准备计算
with tf.compat.v1.Session() as session:  # 创建一个session并打印输出
    session.run(init)  # 初始化变量
    print(session.run(loss))  # 打印损失值

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)  # 结果得到了一个Tensor类型的变量，没有维度，数字类型为int32。之前所做的一切都只是把这些东西放到了一个“计算图(computation graph)”中，还没有开始运行这个计算图
sess = tf.compat.v1.Session()  # 为了实际计算a*b，我们需要创建一个会话并运行它
print(sess.run(c))

# 利用feed_dict来改变x的值
x = tf.compat.v1.placeholder(tf.int64, name="x")  # 当我们第一次定义x时，不必为它指定一个值。 占位符只是一个变量，我们会在运行会话时将数据分配给它。
print(sess.run(2 * x, feed_dict={x: 3}))  # 将x的值设为3，运行2*3
sess.close()


# 线性函数
def linear_function():
    """
    实现一个线性功能：
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    返回：
        result - 运行了session后的结果，运行的是Y = WX + b

    """

    np.random.seed(1)  # 指定随机种子
    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)
    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul是矩阵乘法
    # Y = tf.matmul(W,X) + b #也可以以写成这样子
    # 创建一个session并运行它
    s
    result = sess.run(Y)
    # session使用完毕，关闭它
    sess.close()

    return result


print("result = " + str(linear_function()))  # 测试linear_function()


# 计算sigmoid
def sigmoid(z):
    """
    实现使用sigmoid函数计算z

    参数：
        z - 输入的值，标量或矢量

    返回：
        result - 用sigmoid计算z的值

    """
    # 创建一个占位符x，名字叫“x”
    x = tf.compat.v1.placeholder(tf.float32, name="x")
    # 计算sigmoid(z)
    sigmoid = tf.sigmoid(x)
    # 创建一个会话，使用方法二
    with tf.compat.v1.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result


print("sigmoid(0) = " + str(sigmoid(0)))  # 测试sigmoid()
print("sigmoid(12) = " + str(sigmoid(12)))


# 取一个标签矢量和C类总数，返回一个独热编码
def one_hot_matrix(lables, C):
    """
    创建一个矩阵，其中第i行对应第i个类号，第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签，那么entry (i,j)将会是1

    参数：
        lables - 标签向量
        C - 分类数

    返回：
        one_hot - 独热矩阵

    """

    # 创建一个tf.constant，赋值为C，名字叫C
    C = tf.constant(C, name="C")
    # 使用tf.one_hot，注意一下axis
    one_hot_matrix = tf.one_hot(indices=lables, depth=C,
                                axis=0)  # 如果indices是一个长度为features的向量,axis=0则输出一个depth*features形状的张量
                                         # 如果indices是一个形状为[batch, features]的矩阵,axis=0则输出一个depth * batch * features形状的张量
    # 创建一个session
    sess = tf.compat.v1.Session()
    # 运行session
    one_hot = sess.run(one_hot_matrix)
    # 关闭session
    sess.close()

    return one_hot


labels = np.array([1, 2, 3, 0, 2, 1])  # 测试one_hot_matrix()
one_hot = one_hot_matrix(labels, C=4)  # one_hot为4*6
print(str(one_hot))


# 初始化为0/1
def ones(shape):
    """
    创建一个维度为shape的变量，其值全为1

    参数：
        shape - 你要创建的数组的维度

    返回：
        ones - 只包含1的数组
    """
    # 使用tf.ones()
    ones = tf.ones(shape)
    # 创建会话
    sess = tf.compat.v1.Session()
    # 运行会话
    ones = sess.run(ones)
    # 关闭会话
    sess.close()

    return ones


print("ones = " + str(ones([3])))  # 测试ones()

