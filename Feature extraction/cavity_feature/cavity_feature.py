import cv2
import numpy as np
import os
import pandas as pd


# 空腔圆形度的计算
def calculateCircularity(contour):        # 循环度的计算
    perimeter = cv2.arcLength(contour, True)  # 求周长
    area = cv2.contourArea(contour)     # 求面积
    circularity = 4 * np.pi * area / (perimeter**2)  # 圆
    return circularity

# 空腔轮廓粗糙度的计算
def calculateRoughness(contour):               #粗糙度的计算
    perimeter = cv2.arcLength(contour, True)   # 求周长
    hull = cv2.convexHull(contour)             # 求凸包
    hull_perimeter = cv2.arcLength(hull, True)   # 求凸包的边长
    roughness = perimeter / hull_perimeter
    return roughness

def Image_preprocessing(image_path):
    protein_image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(protein_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    protein_image_gray1 = cv2.bitwise_and(protein_image_gray, protein_image_gray, mask=binary_image)
    protein_image_gray2 = cv2.resize(protein_image_gray1, (1024, 1024))
    binary_image1 = cv2.resize(binary_image, (1024, 1024))


    return protein_image_gray2, binary_image1


def generate_binary_image(size=1024, circle_diameter=760):
    # 创建一个黑色的图像
    image = np.zeros((size, size), np.uint8)
    # 找到图像的中心坐标
    center = (size // 2, size // 2)
    # 画一个白色的圆
    radius = circle_diameter // 2
    color = 255  # 白色
    thickness = -1  # 实心圆
    cv2.circle(image, center, radius, color, thickness)

    return image

def find_max_contours(binary_image):

    contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(contours) != 0:
        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 创建一个与原始图像相同大小的新图像
        result_image = np.zeros_like(binary_image)
        # 将最大轮廓内的像素值设为白色
        cv2.drawContours(result_image, [max_contour], -1, (255), thickness=cv2.FILLED)
    else:
        result_image = np.zeros_like(binary_image)
    return result_image

def get_connected_regions(binary_image, min_area_threshold):
    # 进行连通组件标记
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    # 创建一个和输入图像一样大小的全黑图像
    result_image = np.zeros_like(binary_image)
    # 遍历所有标签，根据面积阈值筛选出成片的大白色区域
    for label in range(1, num_labels):  # 跳过背景标签0
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            # 将面积大于阈值的区域保留为白色
            result_image[labels == label] = 255

    return result_image


def is_binary_image(image):
    # 1. 检查通道数
    if len(image.shape) != 2:  # 如果图像不是单通道的，那么它不是二值图像
        return False

    # 2. 检查像素值范围
    unique_values = np.unique(image)
    if len(unique_values) != 2 or not (0 in unique_values and 255 in unique_values):
        return False
    return True

def get_internal_cavity_area(inner_contour_image):

    fushi_1 = cv2.erode(inner_contour_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=1)
    gaoshi_1 = cv2.GaussianBlur(fushi_1, (5, 5), 0)
    _, fushi_2 = cv2.threshold(gaoshi_1, 127, 255, cv2.THRESH_BINARY)
    penzhang_1 = cv2.dilate(fushi_2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    contours = cv2.findContours(penzhang_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(penzhang_1, [contour], -1, (0), thickness=cv2.FILLED)
    ret, binary = cv2.threshold(penzhang_1, 180, 255, cv2.THRESH_BINARY)
    contours_2 = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for contour in contours_2:
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(binary, [contour], -1, (0), thickness=cv2.FILLED)

    penzhang_2 = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    smoothed_image = cv2.GaussianBlur(penzhang_2, (5, 5), 0)
    _, smoothed_binary = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)

    # cv2.imshow("1_fushi_1", fushi_1)
    # cv2.imshow("2_gaoshi_1", gaoshi_1)
    # cv2.imshow("3_fushi_2", fushi_2)
    # cv2.imshow("4_penzhang_1", penzhang_1)
    # cv2.imshow("5_penzhang_2", penzhang_2)
    # cv2.imshow("6_binary", binary)
    # cv2.imshow("7_penzhang_2", penzhang_2)
    # cv2.imshow("8_smoothed_image", smoothed_image)
    # cv2.imshow("9_smoothed_binary", smoothed_binary)
    # cv2.waitKey()

    return smoothed_binary

image_name = []
category = []
cavity_Number = []                       # 空腔的个数
cavity_area = []                         # 空腔的总面积
cavity_area_mean = []                    # 空腔的面积均值
cavity_area_tario_to_MAXcontour = []     # 空腔总面积与最大轮廓形成的面积占比
cavity_length = []                       # 空腔的总周长
cavity_length_mean = []                  # 空腔的周长均值
cavity_length_tario_to_MAXcontour = []   # 空腔最大周长与最大轮廓形成的占比
circularities = []                       # 空腔轮廓圆形度均值
Roughness = []                           # 空腔轮廓粗糙度均值
dimensions_fractual = []                 # 空腔前景的分形维数
MAX_fractual = []                        # 最大轮廓形成的前景的分形维数
MAX_area = []                            # 最大轮廓形成的前景面积
n = 0

folder_path = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\Process_dataset\\protein"

sub_folders = os.listdir(folder_path)
for sub_folder in sub_folders:
    sub_folder_path = os.path.join(folder_path, sub_folder)
    Images = os.listdir(sub_folder_path)
    for Image_file in Images:
        Image_file_path = os.path.join(sub_folder_path, Image_file)
        image, mark = Image_preprocessing(Image_file_path)

        # MAX_contours_image = find_max_contours(mark)
        MAX_contours_image = generate_binary_image()

        # cv2.imshow("MAX_contours_imagezz", mark)
        # cv2.imshow("MAX_contours_image", MAX_contours_image)
        # cv2.imshow("yuan_image", yuan_image)
        # cv2.waitKey()

        inner_contour_image = cv2.bitwise_xor(mark, MAX_contours_image)

        internal_cavity_image = get_internal_cavity_area(inner_contour_image)
        n += 1
        print(f"数量：{n}，当前处理图像：{Image_file}，亚细胞种类：{sub_folder}")
        # cv2.imshow("image", image)
        # cv2.imshow("internal_cavity_image", internal_cavity_image)
        # cv2.waitKey()

        # inn_cavity_Number = []
        inn_cavity_area = []
        inn_cavity_length = []
        inn_cavity_circularities = []
        inn_cavity_Roughness = []
        m = 0
        contours = cv2.findContours(internal_cavity_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        for contour in contours:
            # inn_cavity_Number.append(len(contours))
            m += 1
            inn_cavity_area.append(cv2.contourArea(contour))
            inn_cavity_length.append(cv2.arcLength(contour, True))
            inn_cavity_circularities.append(calculateCircularity(contour))
            inn_cavity_Roughness.append(calculateRoughness(contour))

        contours1 = cv2.findContours(MAX_contours_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        if len(contours1) != 0:
            MAX_contours_image_area = cv2.contourArea(contours1[0])
            MAX_contours_imag_length = cv2.arcLength(contours1[0], True)
        else:
            MAX_contours_image_area = 0
            MAX_contours_imag_length = 0

        if len(inn_cavity_length) > 0 and MAX_contours_imag_length != 0:
            cavity_area_tario_to_MAXcontour_calculate = np.sum(inn_cavity_area) / MAX_contours_image_area
            cavity_length_tario_to_MAXcontour__calculate = np.max(inn_cavity_length) / MAX_contours_imag_length

            image_name.append(Image_file)
            category.append(sub_folder)
            cavity_Number.append(m)
            cavity_area.append(np.sum(inn_cavity_area))
            cavity_area_mean.append(np.mean(inn_cavity_area))
            cavity_area_tario_to_MAXcontour.append(cavity_area_tario_to_MAXcontour_calculate)
            cavity_length.append(np.sum(inn_cavity_length))
            cavity_length_mean.append(np.mean(inn_cavity_length))
            cavity_length_tario_to_MAXcontour.append(cavity_length_tario_to_MAXcontour__calculate)
            circularities.append(np.mean(inn_cavity_circularities))
            Roughness.append(np.mean(inn_cavity_Roughness))
            # dimensions_fractual.append(fractual(internal_cavity_image))
            # MAX_fractual.append(fractual(MAX_contours_image))
            dimensions_fractual.append(0)
            MAX_fractual.append(0)
            MAX_area.append(MAX_contours_image_area)
        else:
            image_name.append(Image_file)
            category.append(sub_folder)
            cavity_Number.append(0)
            cavity_area.append(np.sum(0))
            cavity_area_mean.append(np.mean(0))
            cavity_area_tario_to_MAXcontour.append(0)
            cavity_length.append(np.sum(0))
            cavity_length_mean.append(np.mean(0))
            cavity_length_tario_to_MAXcontour.append(0)
            circularities.append(0)
            Roughness.append(0)
            dimensions_fractual.append(0)
            # MAX_fractual.append(fractual(MAX_contours_image))
            MAX_fractual.append(0)
            MAX_area.append(MAX_contours_image_area)

        number = range(len(cavity_Number))
        def pd_toexcel(number, image_name, category, cavity_Number, cavity_area, cavity_area_mean, cavity_area_tario_to_MAXcontour, cavity_length,
                       cavity_length_mean, cavity_length_tario_to_MAXcontour, circularities, Roughness, dimensions_fractual,
                       MAX_fractual, MAX_area, filename):  # pandas库储存数据到excel
            dfData = {  # 用字典设置DataFrame所需数据
                '序号': number,
                '图像名称': image_name,
                '亚细胞种类': category,
                '空腔的个数': cavity_Number,
                '空腔总面积': cavity_area,
                '空腔面积均值': cavity_area_mean,
                '空腔总面积与最大轮廓形成的面积占比': cavity_area_tario_to_MAXcontour,
                '空腔总周长': cavity_length,
                '空腔周长均值': cavity_length_mean,
                '空腔最大周长与最大轮廓形成的占比': cavity_length_tario_to_MAXcontour,
                '空腔轮廓圆形度均值': circularities,
                '空腔轮廓粗糙度均值': Roughness,
                '空腔前景的分形维数': dimensions_fractual,
                '最大轮廓形成的前景的分形维数': MAX_fractual,
                '最大轮廓形成的前景面积': MAX_area
            }
            df = pd.DataFrame(dfData)  # 创建DataFrame
            df.to_excel(filename, index=False)  # 存表，去除原始索引列（0,1,2...）

        pd_toexcel(number, image_name, category, cavity_Number, cavity_area, cavity_area_mean, cavity_area_tario_to_MAXcontour, cavity_length, cavity_length_mean,
                   cavity_length_tario_to_MAXcontour, circularities, Roughness, dimensions_fractual, MAX_fractual, MAX_area,
                   'E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Feature_(cavity_feature_REprotein).xlsx')


