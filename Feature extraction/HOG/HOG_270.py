import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import hog
from skimage import data, exposure


def Image_preprocessing(image_path):
    protein_image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(protein_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    protein_image_gray1 = cv2.bitwise_and(protein_image_gray, protein_image_gray, mask=binary_image)
    protein_image_gray1 = cv2.resize(protein_image_gray1, (3000, 3000))

    return protein_image_gray1


def calculate_HOG(folder_path):
    dict_HOGs = {}
    sub_folders = os.listdir(folder_path)
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue
        Images = os.listdir(sub_folder_path)
        for Image_file in Images:
            Image_file_path = os.path.join(sub_folder_path, Image_file)
            image = Image_preprocessing(Image_file_path)
            fd, hog_image = hog(image,
                                orientations=9,
                                pixels_per_cell=(16, 16),
                                cells_per_block=(2, 2),
                                visualize=True,
                                transform_sqrt=True,
                                multichannel=False)  # multichannel=True是针对3通道彩色；
            print(f"正常处理：{Image_file}，{sub_folder}")
            dict_HOGs[(Image_file, sub_folder)] = fd
    return dict_HOGs

def pd_toexcel(data_dict, filename):
    # 创建一个空的 DataFrame
    df = pd.DataFrame()
    # 遍历字典中的每个键值对，按照您的格式将数据添加到 DataFrame 中
    for key, values in data_dict.items():
        image_name, subcellular_type = key
        df.at[len(df), '图像名称'] = image_name
        df.at[len(df)-1, '亚细胞种类'] = subcellular_type
        for j in range(len(values)):
            df.at[len(df) - 1, f'HOG{j+1}'] = values[j]
        # 将 DataFrame 写入到 Excel 文件中
    df.to_excel(filename, index=False)


if __name__== '__main__':
    folder_path = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map2\\Image_80x80"
    outer_path = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Deep2\\Feature_(HOG_270+++).xlsx"
    dict_HOGs =calculate_HOG(folder_path)
    pd_toexcel(dict_HOGs, outer_path)





