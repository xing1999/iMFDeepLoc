import cv2
import os
import numpy as np
import pandas as pd


def calculate_co_localization_ratio(dna_folder, protein_folder, subfolders):
    co_localization_ratios = {}  # 存储共定位比率的字典

    # 遍历所有亚细胞类别
    for subfolder in subfolders:
        dna_subfolder_path = os.path.join(dna_folder, subfolder)
        protein_subfolder_path = os.path.join(protein_folder, subfolder)

        # 获取DNA文件夹下的所有图像文件名
        dna_images = os.listdir(dna_subfolder_path)

        # 遍历DNA文件夹下的每个图像
        for dna_image in dna_images:
            # 构建DNA和蛋白质图像的完整路径
            dna_image_path = os.path.join(dna_subfolder_path, dna_image)
            protein_image_path = os.path.join(protein_subfolder_path, dna_image)

            # 检查蛋白质图像是否存在
            if not os.path.exists(protein_image_path):
                print(f"Warning: Protein image not found for {dna_image} in {subfolder}. Skipping...")
                continue

            # 读取DNA和蛋白质图像
            dna_image_gray = cv2.imread(dna_image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(dna_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dna_image_gray1 = cv2.bitwise_and(dna_image_gray, dna_image_gray, mask=binary_image)
            dna_image_gray2 = cv2.resize(dna_image_gray1, (1024, 1024))

            protein_image_gray = cv2.imread(protein_image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image1 = cv2.threshold(protein_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            protein_image_gray1 = cv2.bitwise_and(protein_image_gray, protein_image_gray, mask=binary_image1)
            protein_image_gray2 = cv2.resize(protein_image_gray1, (1024, 1024))

            # 计算共定位像素数量
            co_localization_pixels = np.sum((dna_image_gray2 > 0) & (protein_image_gray2 > 0))
            # 计算蛋白质片段中的像素数量
            protein_pixels = np.sum(protein_image_gray2 > 0)
            # 计算DNA片段中的像素数量
            DNA_pixels = np.sum(dna_image_gray2 > 0)
            # 计算共定位比率并添加到字典中
            co_localization_ratio_protein = co_localization_pixels / protein_pixels if protein_pixels > 0 else 0
            co_localization_ratio_DNA = co_localization_pixels / DNA_pixels if DNA_pixels > 0 else 0

            # 计算与DNA区域共定位的蛋白质片段中的像素值总和
            co_localization_sum1 = np.sum(np.where((dna_image_gray2 > 0) & (protein_image_gray2 > 0), protein_image_gray2, 0))
            co_localization_sum2 = np.sum(np.where((dna_image_gray2 > 0) & (protein_image_gray2 > 0), dna_image_gray2, 0))
            # 计算与 DNA 区域共定位的蛋白质片段中的像素值总和与蛋白质/DNA片段中的像素总和的比率
            co_localization_value_ratio_protein = co_localization_sum1 / np.sum(protein_image_gray2) if np.sum(protein_image_gray2) > 0 else 0
            co_localization_value_ratio_DNA = co_localization_sum2 / np.sum(dna_image_gray2) if np.sum(dna_image_gray2) > 0 else 0

            co_localization_ratios[(dna_image, subfolder)] = [co_localization_ratio_protein, co_localization_ratio_DNA, co_localization_value_ratio_protein, co_localization_value_ratio_DNA]
            print(f"图像名称:{dna_image}； 亚细胞种类：{subfolder}；共同像素点与蛋白质像素点的数量比值：{co_localization_ratio_protein}；共同像素点与DNA像素点的数量比值：{co_localization_ratio_DNA}；共同像素点与蛋白质像素点灰度值的比值：{co_localization_value_ratio_protein}；共同像素点与DNA像素点灰度值的比值：{co_localization_value_ratio_DNA}")

    return co_localization_ratios


def pd_toexcel(data_dict, filename):
    # 创建一个空的 DataFrame
    df = pd.DataFrame()
    # 遍历字典中的每个键值对，按照您的格式将数据添加到 DataFrame 中
    for key, values in data_dict.items():
        image_name, subcellular_type = key
        df.at[len(df), '图像名称'] = image_name
        df.at[len(df)-1, '亚细胞种类'] = subcellular_type
        df.at[len(df)-1, '共同像素点与蛋白质像素点的"数量"比值'] = values[0]
        df.at[len(df)-1, '共同像素点与DNA像素点的"数量"比值'] = values[1]
        df.at[len(df)-1, '共同像素点与蛋白质像素点"灰度值"的比值'] = values[2]
        df.at[len(df)-1, '共同像素点与DNA像素点"灰度值"的比值'] = values[3]
    # 将 DataFrame 写入到 Excel 文件中
    df.to_excel(filename, index=False)


# 示例用法
dna_folder = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\Process_dataset\\DNA"
protein_folder = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\Process_dataset\\protein"
subfolders = ['Cy', 'Er', 'Go', 'Mi', 'Nu', 'Ve']
co_localization_ratios = calculate_co_localization_ratio(dna_folder, protein_folder, subfolders)
# 打印共定位比率
# for (image_name, subfolder), ratio in co_localization_ratios.items():
#     print(f"Image: {image_name}, Subfolder2: {subfolder}, Co-localization Ratio: {ratio}")
pd_toexcel(co_localization_ratios, 'E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Feature_(tario).xlsx')
