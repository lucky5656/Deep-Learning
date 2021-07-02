import os
import PIL.Image as Image


def changeJpgToPng(w, h, path):
    # 修改图像大小
    image = Image.open(path)
    image = image.resize((w, h), Image.ANTIALIAS)

    # 将jpg转换为png
    png_name = str(path)[0:-len('.jpg')] + '.png'
    image.save(png_name)

    print(png_name)

    # 去白底
    image = image.convert('RGB')
    img_w, img_h = image.size
    color_white = (255, 255, 255)
    for j in range(img_h):
        for i in range(img_w):
            pos = (i, j)
            color_now = image.getpixel(pos)
            if color_now == color_white:
                # 透明度置为0
                color_now = color_now[:-1] + (0,)
                image.putpixel(pos, color_now)
    image.save(png_name)


if __name__ == '__main__':
    t_w = 64
    t_h = 64

    dic = os.listdir('image')
    for d in dic:
        if d.count('.jpg') > 0:
            changeJpgToPng(t_w, t_h, 'image/' + d)