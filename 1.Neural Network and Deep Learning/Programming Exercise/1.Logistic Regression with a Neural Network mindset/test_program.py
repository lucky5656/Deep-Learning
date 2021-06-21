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
    k = line.split(' ')[0]
    v = line.split(' ')[1]
    dict_temp[k] = v
# 依旧是关闭文件
f.close()
print(dict_temp)


'# 测试自己的图片'
## START CODE HERE ## (PUT YOUR IMAGE NAME)
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