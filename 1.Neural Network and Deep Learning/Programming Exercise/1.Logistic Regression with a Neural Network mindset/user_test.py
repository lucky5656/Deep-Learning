###运行不了,txt文件还原为为字典未实现。

import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.transform import resize

# 声明一个空字典，来保存文本文件数据
dict_temp = {}
# 打开文本文件
file = open('模型参数.txt','r')
# 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
for line in file.readlines():
    line = line.strip()
    k = line.split(':')[0]
    v = line.split(':')[1]
    dict_temp[k] = v
# 依旧是关闭文件
file.close()
print(dict_temp)

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def predict(w, b, X):
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

'# 测试自己的图片'
## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "16.jpg"  # change this to the name of your image file

# We preprocess the image to fit your algorithm.
fname = "C:/Users/123/Pictures/" + my_image
image = np.array(imageio.imread(fname))

num_px = 64
my_image = resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
# my_image = my_image / 255.
my_predicted_image = predict(dict_temp["w"], dict_temp["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \""
      + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
plt.show()