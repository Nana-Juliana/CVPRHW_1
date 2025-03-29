from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
import tkinter
import numpy as np
import cv2 as cv
import cv2
import pytesseract
from numpy import *


def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.LANCZOS)


def clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


root = Tk()
root.geometry('900x600')
root.title("户型识别")

fileString = ""


def xz():
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        load = Image.open(filename)
        w, h = load.size
        resized = resize(w, h, 270, 270, load)

        render = ImageTk.PhotoImage(resized)
        img = tkinter.Label(image=render, width=270, height=270)
        img.image = render
        img.place(x=10, y=30)

        lb.config(text="");

        global fileString
        fileString = filename
        print(fileString)
    else:
        lb.config(text="您没有选择任何文件");


def run():
    global fileString
    if fileString == "":
        print("请选择文件")
        return

    print(fileString)
    src = cv.imread(fileString)

    # if fileString == "C:/Users/Administrator/Desktop/p1.png":
    if fileString == "D:/Pictures/P1.png":
        cv.line(src, (108, 240), (145, 240), (0, 0, 0), 8)  # 9
    cv.imwrite("to_mix.png", src.copy())
    #     cv.imshow("src",src)

    ###################比例尺
    #     src_2 = src.copy()

    # HSV thresholding to get rid of as much background as possible
    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))

    # Adaptive Thresholding to isolate the bed
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the rectangle by choosing only the big ones
    # and choose the brightest rectangle as the bed
    max_brightness = 0
    canvas = src.copy()
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w * h > 40000 and x > 0:
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y + h, x:x + w] = src[y:y + h, x:x + w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
                #         cv2.imshow("mask", mask)

    x1, y1, w, h = brightest_rectangle
    x2 = x1 + w
    y2 = y1 + h
    print(x1, y1, x2, y2)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 4)
    #     cv2.imshow("canvas", canvas)
    #     cv2.imwrite("result.jpg", canvas)
    ######################################################
    (ruler_h, ruler_w, o) = src.shape
    ruler_topsrc = src[0:y1, 0:ruler_w]
    cv2.imshow("ruler_top", ruler_topsrc)

    ret, ruler_top = cv2.threshold(ruler_topsrc, 200, 255, cv2.THRESH_BINARY_INV)
    ruler_top = cv2.cvtColor(ruler_top, cv2.COLOR_BGR2GRAY)

    Mat_top = ruler_top.copy()

    A_top = [0 for z in range(0, ruler_w)]

    for i in range(0, ruler_w):
        for j in range(0, y1):
            if ruler_top[j, i] > 100:
                A_top[i] += 1
                Mat_top[j, i] = 0
    for i in range(0, ruler_w):
        for j in range(0, A_top[i]):
            Mat_top[j, i] = 255  # 设置黑点
    cv2.imshow("Mat_top", Mat_top)

    #####统计分界线段位置
    divide = []
    for x in range(0, ruler_w):
        if A_top[x] > 20:
            print(x)
            divide.append(x)
            for x2 in range(x + 1, x + 3):
                if A_top[x2] > 20:
                    A_top[x2] = 0
    ####
    print(divide)

    #if fileString == "C:/Users/Administrator/Desktop/p1.png":
    if fileString == "D:/Pictures/P1.png":
        rulerarray = []

        number = src[0:y1, divide[0]:divide[1]]
        # name = "number"+str(x+1)
        # cv2.imshow(name, number)
        content = pytesseract.image_to_string(number)  # 解析图片
        ruler = round(int(content) / (divide[1] - divide[0]) * 10) / 10
        rulerarray.append(ruler)

        number = src[0:y1, divide[1]:divide[2]]
        # name = "number"+str(x+1)
        # cv2.imshow(name, number)
        content = pytesseract.image_to_string(number)  # 解析图片
        ruler = round(int(content) / (divide[2] - divide[1]) * 10) / 10
        rulerarray.append(ruler)

        number = src[0:y1, divide[2]:divide[3]]
        # name = "number"+str(x+1)
        # cv2.imshow(name, number)
        content = pytesseract.image_to_string(number)  # 解析图片
        ruler = round(int(content) / (divide[3] - divide[2]) * 10) / 10
        rulerarray.append(ruler)

        number = src[0:y1, divide[3]:divide[4]]
        # name = "number"+str(x+1)
        # cv2.imshow(name, number)
        content = pytesseract.image_to_string(number)  # 解析图片
        ruler = round(int(content) / (divide[4] - divide[3]) * 10) / 10
        rulerarray.append(ruler)

        print(rulerarray)
        ruler_final = round(mean(rulerarray) * 10) / 10
        print(ruler_final)
    else:
        ruler_final = 25

    lb5.config(text=ruler_final);

    #######################################预处理
    dst1 = cv.fastNlMeansDenoisingColored(src, None, 10, 10, 7, 21)
    # cv2.imshow('GRAY_1',dst1)

    dst2 = cv.fastNlMeansDenoisingColored(dst1, None, 10, 10, 7, 21)
    # cv2.imshow('GRAY_2',dst2)

    grayImage = cv.cvtColor(dst2, cv.COLOR_BGR2GRAY)
    # cv2.imshow('GRAY',grayImage)

    kernel = np.ones((3, 3), np.uint8)
    blur = cv.GaussianBlur(grayImage, (5, 5), 0)
    # cv2.imshow("blur",blur)

    open1 = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)
    dst = cv.morphologyEx(open1, cv.MORPH_CLOSE, kernel)
    # cv2.imshow('dst',dst)

    ret, thresh1 = cv.threshold(dst, 45, 255, cv.THRESH_BINARY)
    #     cv.imshow("imgth", thresh1)

    cv.imwrite("line_a.png", thresh1)
    ###################################################显示图片2

    load = Image.open("line_a.png")
    w, h = load.size
    resized = resize(w, h, 270, 270, load)

    render = ImageTk.PhotoImage(resized)
    img = tkinter.Label(image=render, width=270, height=270)
    img.image = render
    img.place(x=330, y=20)

    #####################################################

    img = cv.imread("line_a.png")
    # img = thresh1
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img1 = cv.Canny(gray, 50, 150, apertureSize=3)
    cv.imshow('edges', img1)

    # 返回图像的高和宽
    (h, w) = img1.shape
    print("h,w", h, " ", w)
    Mat1 = img1.copy()

    # 初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
    a = [0 for z in range(0, w)]

    for i in range(0, w):
        for j in range(0, h):
            if img1[j, i] == 255:  # 判断该点是否为黑点，0代表是黑点
                a[i] += 1  # 该列的计数器加1
                Mat1[j, i] = 0  # 记录完后将其变为白色，即等于255
    for i in range(0, w):
        for j in range(h - a[i], h):  # 从该列应该变黑的最顶部的开始向最底部设为黑点
            Mat1[j, i] = 255  # 设为黑点

    # cv.imshow("Mat1",Mat1)

    img2 = img1.copy()

    flag = 1

    # 纵向墙体的校正
    for x in range(0, w):
        if a[x] > 50:
            #         print(x)
            for y in range(x - 5, x + 5):
                if a[x] < a[y]:
                    flag = 0
            if flag == 1:
                for i in range(x - 5, x + 5):
                    for j in range(0, h):
                        if img2[j, i] > 0:
                            img2[j, i] = 0
                            img2[j, x] = 255

    # 删除横向墙体的像素
    for x in range(0, w):
        if a[x] < 50:
            for j in range(0, h):
                if img2[j, x] > 0:
                    img2[j, x] = 0

    # 排除竖切横向墙体得到的点的干扰
    for x in range(0, w):
        if a[x] > 50:
            for j in range(0, h):
                if img2[j, x] > 0 and img2[j + 1, x] == 0 and img2[j - 1, x] == 0:
                    for z1 in range(j + 2, j + 30):  # 不能把该点本身计算在内
                        if img2[z1, x] > 0 and img2[z1 + 1, x] == 0 and img2[z1 - 1, x] == 0:
                            img2[z1, x] = 0
                            img2[j, x] = 0
    #
    # 离散的点相连
    for x in range(0, w):
        if a[x] > 50:
            for j in range(0, h):
                if img2[j, x] > 0 and img2[j + 1, x] == 0:
                    for z1 in range(j, j + 30):
                        if img2[z1, x] > 0:
                            for z2 in range(j, z1):
                                img2[z2, x] = 255
    # 墙体两条线转化为一条墙中心线
    cv.imshow("img2", img2)
    # img3 = img2.copy()
    img3 = np.zeros((h, w, 1), dtype=np.uint8)
    for x in range(0, w):
        if a[x] > 50:
            for y in range(x + 1, x + 20):
                if a[y] > 50:
                    mid = int(round((x + y) / 2))
                    for j in range(0, h):
                        if img2[j, x] == img2[j, y] and img2[j, y] > 0:
                            img3[j, mid] = 255
    #                 mid = int((x+y)/2)
    #                 for i in range(x-1,y+1):
    #                     for j in range(0,h):
    #                         if img2[j,i]>0:
    # #                             img3[j,i]=0
    #                             img3[j,mid]=255

    # 基于中心线 显示可能的连线
    b = [0 for z in range(0, w)]
    for i in range(0, w):
        for j in range(0, h):
            if img3[j, i] == 255:
                b[i] += 1

    for x in range(0, w):
        if b[x] > 50:
            #         print(x)
            for j in range(0, h):
                if img3[j, x] > 0 and img3[j + 1, x] == 0:
                    for z1 in range(j, j + 50):
                        if img3[z1, x] > 0:
                            for z2 in range(j, z1):
                                img3[z2, x] = 255

    cv.imshow("img3", img3)

    MatI = img1.copy()

    # 初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
    A = [0 for z in range(0, h)]

    for i in range(0, h):
        for j in range(0, w):
            if img1[i, j] == 255:
                A[i] += 1
                MatI[i, j] = 0
                # for i in range(0,h):
    #     for j in range(w-a[i],w):  #从该列应该变黑的最顶部的开始向最底部设为黑点
    #         MatI[i,j]=255            #设为黑点

    for i in range(0, h):  # 遍历每一行
        for j in range(0, A[i]):  # 从该行应该变黑的最左边的点开始向最右边的点设置黑点
            MatI[i, j] = 255  # 设置黑点

    # cv.imshow("MatI",MatI)

    imgII = img1.copy()

    flagI = 1

    # 横向墙体的校正
    for x in range(0, h):
        if A[x] > 50:
            #         print(x)
            for y in range(x - 5, x + 5):
                if A[x] < A[y]:
                    flagI = 0
                if flagI == 1:
                    for i in range(x - 5, x + 5):
                        for j in range(0, w):
                            if imgII[i, j] > 0:
                                #                             count =count+1
                                imgII[i, j] = 0
                                imgII[x, j] = 255

    # 删除纵向墙体的像素
    for x in range(0, h):
        if A[x] < 50:
            for j in range(0, w):
                if imgII[x, j] > 0:
                    imgII[x, j] = 0

    # #排除竖切横向墙体得到的点的干扰|
    for x in range(0, h):
        if A[x] > 50:
            for j in range(0, w):
                if imgII[x, j] > 0 and imgII[x, j + 1] == 0 and (
                        imgII[x, j - 1] or imgII[x, j - 2] or imgII[x, j - 3] or imgII[x, j - 4] or imgII[
                    x, j - 5]) == 0:
                    for z1 in range(j + 2, j + 50):  # 不能把该点本身计算在内
                        if imgII[x, z1] > 0 and imgII[x, z1 - 1] == 0 and (
                                imgII[x, z1 + 1] or imgII[x, z1 + 2] or imgII[x, z1 + 3] or imgII[x, z1 + 4] or imgII[
                            x, z1 + 5]) == 0:
                            imgII[x, j] = 0
                            imgII[x, z1] = 0

    # 离散的点相连
    for x in range(0, h):
        if A[x] > 50:
            for j in range(0, w):
                if imgII[x, j] > 0 and imgII[x, j + 1] == 0:
                    for z1 in range(j, j + 30):
                        if imgII[x, z1] > 0:
                            for z2 in range(j, z1):
                                imgII[x, z2] = 255

    # 墙体两条线转化为一条墙中心线
    #     cv.imshow("imgII",imgII)
    # imgIII = imgII.copy()
    imgIII = np.zeros((h, w, 1), dtype=np.uint8)
    for x in range(0, h):
        if A[x] > 50:
            for y in range(x + 1, x + 10):
                if A[y] > 50:
                    mid = int(round((x + y) / 2))
                    for j in range(0, w):
                        if imgII[x, j] == imgII[y, j] and imgII[y, j] > 0:
                            imgIII[mid, j] = 255

    # 基于中心线 显示可能的连线
    B = [0 for z in range(0, h)]
    for i in range(0, h):
        for j in range(0, w):
            if imgIII[i, j] == 255:
                B[i] += 1

    for x in range(0, h):
        if B[x] > 30:
            #             print(x)
            for j in range(0, w):
                if imgIII[x, j] > 0 and imgIII[x, j + 1] == 0:
                    if (j + 150) > w:
                        limit = w - j
                    else:
                        limit = j + 150
                    for z1 in range(j + 1, limit):
                        if imgIII[x, z1] > 0:
                            for z2 in range(j, z1):
                                imgIII[x, z2] = 255

    img4 = img3 + imgIII
    #     cv.imshow("img4",img4)

    cv.imwrite("area detect2.png", img4)
    ###################################################显示图片3

    load = Image.open("area detect2.png")
    w, h = load.size
    resized = resize(w, h, 270, 270, load)

    render = ImageTk.PhotoImage(resized)
    img = tkinter.Label(image=render, width=270, height=270)
    img.image = render
    img.place(x=10, y=310)
    ##################################
    src = cv.imread("area detect2.png")
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    #     cv.imshow("binary", binary)
    cv.imwrite('binary.png', binary)

    output = cv.connectedComponents(binary, connectivity=8, ltype=cv.CV_32S)
    num_labels = output[0]
    print(output[1].shape)
    print(num_labels)  # output: 5
    lables = output[1]

    count = []
    for i in range(num_labels):
        count.append(0)
        for row in range(h):
            for col in range(w):
                if lables[row, col] == i:
                    count[i] += 1
    print(count)  ##得到每个区域的像素点数量

    #########################################
    ruler_real = (ruler_final / 1000) * (ruler_final / 1000)
    for i in range(num_labels):
        if i > 1:
            x_room = 730
            y_room = 330 + 30 * (i - 2)
            str_room = "room" + str(i - 1) + " : " + str(round(count[i] * ruler_real * 10) / 10)
            lb_room = Label(root, text=str_room)
            lb_room.place(x=x_room, y=y_room)
    sum = 0
    for i in range(num_labels):
        if i > 1:
            sum += count[i]
    x_room = 730
    y_room = 300
    str_room = "总面积: " + str(round(sum * ruler_real * 10) / 10)
    lb_room = Label(root, text=str_room)
    lb_room.place(x=x_room, y=y_room)

    #####################################################

    # 构造颜色
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))
    colors[0] = (0, 0, 0)
    colors[1] = (255, 255, 255)

    # 画出连通图
    h, w = gray.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[lables[row, col]]

    cv.imshow("colored labels", image)
    cv.imwrite("lables.png", image)
    components = num_labels - 2
    print("total componets : ", components)
    ##显示房间数量
    lb6.config(text=components);

    img_lable = cv.imread('lables.png')
    # h2, w2 = img_lable.shape

    img_src = cv.imread('to_mix.png')
    # # load = Image.open(filename)
    # w, h = img_src.size
    # resized =resize(w, h, 300, 300, load)

    img_lable2 = cv.resize(img_lable, (500, 500), interpolation=cv.INTER_CUBIC)
    img_src2 = cv.resize(img_src, (500, 500), interpolation=cv.INTER_CUBIC)

    img_mix = cv.addWeighted(img_lable2, 0.6, img_src2, 0.4, 0)

    cv.imshow("mix", img_mix)
    cv.imwrite("mix.png", img_mix)
    ###################################################显示图片4

    load = Image.open("mix.png")
    w, h = load.size
    resized = resize(w, h, 270, 270, load)

    render = ImageTk.PhotoImage(resized)
    img = tkinter.Label(image=render, width=270, height=270)
    img.image = render
    img.place(x=330, y=310)

    #####################################################

    ####################################################

    cv.waitKey(0)
    cv.destroyAllWindows()


#####################
#########################


lb = Label(root, text='')
lb.pack(anchor="n")
lb2 = Label(root, text='原图:')
lb2.place(x=5, y=5)
lb3 = Label(root, text='比例尺系数(mm/pixel):')
lb3.place(x=700, y=100)
lb4 = Label(root, text='房间数量(个):')
lb4.place(x=700, y=180)
lb4 = Label(root, text='户型总面积(m2):')
lb4.place(x=700, y=260)

lb5 = Label(root, text='')
lb5.place(x=730, y=140)
lb6 = Label(root, text='')
lb6.place(x=730, y=220)

# lb7 = Label(root,text = '')
# lb7.place(x=730, y=300)

btn1 = Button(root, text="读取文件", width=20, command=xz)
btn1.place(x=700, y=510)
btn2 = Button(root, text="运行算法", width=20, command=run)
btn2.place(x=700, y=550)

root.mainloop()


