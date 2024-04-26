import cv2
import os
import numpy as np
import pandas as pd


def calculate_gray_mean(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_gray = cv2.bitwise_and(image, image, mask=binary_image)
    gray_resized_image = cv2.resize(image_gray, (1024, 1024))

    non_zero_pixels = cv2.findNonZero(gray_resized_image)
    # 初始化非零像素的灰度值总和和像素数量
    sum_gray_values = 0
    num_non_zero_pixels = 0
    # 遍历非零像素，计算灰度值总和
    for pixel in non_zero_pixels:
        x, y = pixel[0]
        gray_value = gray_resized_image[y, x]  # OpenCV中的像素访问是(y, x)
        sum_gray_values += gray_value
        num_non_zero_pixels += 1
    # 计算灰度均值
    if num_non_zero_pixels > 0:
        mean_gray_value = sum_gray_values / num_non_zero_pixels
    else:
        mean_gray_value = 0
    return mean_gray_value ,sum_gray_values, num_non_zero_pixels


def process_images_in_folders(folder_path):
    DNA_gray_mean = []
    protein_gray_mean = []
    ratio_gray_mean = []
    DNA_gray_ALL = []
    protein_gray_ALL = []
    ratio_gray_ALL = []
    subfolder1s = os.listdir(folder_path)
    for subfolder1 in subfolder1s:
        if subfolder1 == "unmix_composition":
            continue
        subfolder1_path = os.path.join(folder_path, subfolder1)
        subfolder2s = os.listdir(subfolder1_path)
        for subfolder2 in subfolder2s:
            subfolder2_path = os.path.join(subfolder1_path, subfolder2)
            image_files = os.listdir(subfolder2_path)
            for image_file in image_files:
                image_path = os.path.join(subfolder2_path, image_file)
                if os.path.isfile(image_path):
                    mean_gray_value ,sum_gray_values, num_non_zero_pixels = calculate_gray_mean(image_path)
                    if subfolder1 == "DNA":
                        DNA_gray_mean.append([image_file, subfolder2, mean_gray_value])
                        DNA_gray_ALL.append([image_file, subfolder2, sum_gray_values])
                        print(f"Image: {image_file}, Subfolder2: {subfolder2}, Subfolder1: {subfolder1}, Mean Gray Value: {mean_gray_value}")
                    else:
                        protein_gray_mean.append([image_file, subfolder2, mean_gray_value])
                        protein_gray_ALL.append([image_file, subfolder2, sum_gray_values])
                        print(f"Image: {image_file}, Subfolder2: {subfolder2}, Subfolder1: {subfolder1}, Mean Gray Value: {mean_gray_value}")

    # DNA_dict = {item[0]: item for item in DNA_gray_mean}
    protein_dict = {item[0]: item for item in protein_gray_mean}
    for DNA_sample in DNA_gray_mean:
        DNA_sample_name = DNA_sample[0]
        if DNA_sample_name in protein_dict:
            protein_sample = protein_dict[DNA_sample_name]
            ratio_mean = protein_sample[2] / DNA_sample[2]
            ratio_gray_mean.append([DNA_sample_name, DNA_sample[1], ratio_mean])
        else:
            print(f"Warning: No matching file found for DNA sample {DNA_sample_name} in protein samples.")

    # DNA_dict_ALL = {item[0]: item for item in DNA_gray_ALL}
    protein_dict_ALL = {item[0]: item for item in protein_gray_ALL}
    for DNA_sample1 in DNA_gray_ALL:
        DNA_sample_name1 = DNA_sample1[0]
        if DNA_sample_name1 in protein_dict_ALL:
            protein_sample1 = protein_dict_ALL[DNA_sample_name1]
            ratio_mean1 = protein_sample1[2] / DNA_sample1[2]
            ratio_gray_ALL.append([DNA_sample_name1, DNA_sample1[1], ratio_mean1])
        else:
            print(f"Warning: No matching file found for DNA sample {DNA_sample_name1} in protein samples.")

    all_data_grayfeature = {}                        # {('zz.jpg', '亚细胞种类'): [6, 4, 2, 6, 4, 2], ('zzj.jpg', '亚细胞种类'): [6, 4, 2, 6, 4, 2], ('zzk.jpg', '亚细胞种类'): [6, 4, 2, 6, 4, 2]}
    for data_list in [DNA_gray_mean, protein_gray_mean, ratio_gray_mean, DNA_gray_ALL, protein_gray_ALL, ratio_gray_ALL]:
        for item in data_list:
            image_name = item[0]
            folder_name = item[1]
            if (image_name, folder_name) not in all_data_grayfeature:
                all_data_grayfeature[(image_name, folder_name)] = [item[2]]
            else:
                all_data_grayfeature[(image_name, folder_name)].append(item[2])
    return all_data_grayfeature


def pd_toexcel(data_dict, filename):
    # 创建一个空的 DataFrame
    df = pd.DataFrame()
    # 遍历字典中的每个键值对，按照您的格式将数据添加到 DataFrame 中
    for key, values in data_dict.items():
        image_name, subcellular_type = key
        df.at[len(df), '图像名称'] = image_name
        df.at[len(df)-1, '亚细胞种类'] = subcellular_type
        df.at[len(df)-1, 'DNA灰度均值'] = values[0]
        df.at[len(df)-1, '蛋白质灰度均值'] = values[1]
        df.at[len(df)-1, 'DNA与蛋白质灰度均值比值'] = values[2]
        df.at[len(df)-1, 'DNA总像素值'] = values[3]
        df.at[len(df)-1, '蛋白质总像素值'] = values[4]
        df.at[len(df)-1, 'DNA与蛋白质总像素值比值'] = values[5]
    # 将 DataFrame 写入到 Excel 文件中
    df.to_excel(filename, index=False)



folder_path = "data/train100_data/Process_dataset"
all_data_grayfeature = process_images_in_folders(folder_path)
pd_toexcel(all_data_grayfeature, 'Feature1.xlsx')