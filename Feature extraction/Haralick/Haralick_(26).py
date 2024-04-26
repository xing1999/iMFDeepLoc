import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
import pandas as pd
import time

tic = time.time()


# 图像读取处理
def Image_preprocessing(image_path):
    protein_image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(protein_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    protein_image_gray1 = cv2.bitwise_and(protein_image_gray, protein_image_gray, mask=binary_image)
    # protein_image_gray2 = cv2.resize(protein_image_gray1, (1024, 1024))

    return protein_image_gray1


# 从图像中提取哈拉利克纹理特征的函数
def extract_features(image):
    # 计算4种类型邻接的哈拉利克纹理特征
    textures = mt.features.haralick(image)

    # 取其平均值并返回(vertical_horizontal_features是水平和竖直、diagonal_features是两对角线、ht_mean是四个共生矩阵得到的均值)
    vertical_horizontal_features = (textures[0] + textures[2]) / 2
    diagonal_features = (textures[1] + textures[3]) / 2
    ht_mean = textures.mean(axis=0)

    return ht_mean, vertical_horizontal_features, diagonal_features


# 加载训练数据集
train_path = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map2/Image_80x80"  # 输入存储所有图像的目录
train_labels = ['Cy', 'Er', 'Go', 'Mi', 'Nu', 'Ve']  # 假设这些是您的子文件夹名称

# 创建一个空的DataFrame来存储特征
feature_df = pd.DataFrame(columns=['图像名称', '子文件夹'] + [f'Haralick{i}' for i in range(1, 27)])


# 遍历训练数据集
print("[状态] 开始提取哈拉利克纹理特征..")
for label in train_labels:
    cur_path = os.path.join(train_path, label, '*g')
    for file in glob.glob(cur_path):
        print("处理图像 - {} in {}".format(file, label))

        gray = Image_preprocessing(file)
        # 从图像中提取哈拉利克纹理
        mean4_features, vertical_horizontal_features_mean, diagonal_features_mean= extract_features(gray)
        # features = vertical_horizontal_features_mean.extend(diagonal_features_mean)
        features = np.concatenate((vertical_horizontal_features_mean, diagonal_features_mean))
        # 将图像名称、子文件夹名称和特征附加到DataFrame中
        feature_df = feature_df.append({'图像名称': os.path.basename(file), '子文件夹': label,
                                        **{f'Haralick{i}': feature for i, feature in enumerate(features, 1)}},
                                       ignore_index=True)

# 将DataFrame保存为Excel文件
feature_df.to_excel("E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Deep2\\Feature_(Haralick26+++).xlsx", index=False)

toc = time.time()
print("计算时间为 {} 分钟.".format((toc - tic) / 60))
