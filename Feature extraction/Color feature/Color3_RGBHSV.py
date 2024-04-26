import cv2
import os
import numpy as np
import pandas as pd

def calculate_co_localization_ratio(dna_folder, protein_folder, origin_folder, subfolders):
    co_localization_ratios = {}

    # 遍历所有亚细胞类别
    for subfolder in subfolders:
        dna_subfolder_path = os.path.join(dna_folder, subfolder)
        protein_subfolder_path = os.path.join(protein_folder, subfolder)
        origin_folder_path = os.path.join(origin_folder, subfolder)

        # 获取DNA文件夹下的所有图像文件名
        dna_images = os.listdir(dna_subfolder_path)

        # 遍历DNA文件夹下的每个图像
        for dna_image in dna_images:
            # 构建DNA和蛋白质图像的完整路径
            dna_image_path = os.path.join(dna_subfolder_path, dna_image)
            protein_image_path = os.path.join(protein_subfolder_path, dna_image)
            origin_image_path = os.path.join(origin_folder_path, dna_image)

            # 检查蛋白质图像是否存在
            if not os.path.exists(protein_image_path):
                print(f"Warning: Protein image not found for {dna_image} in {subfolder}. Skipping...")
                continue
            if not os.path.exists(origin_image_path):
                print(f"Warning: Protein image not found for {dna_image} in {subfolder}. Skipping...")
                continue
            # 读取DNA和蛋白质图像
            origin_image = cv2.imread(origin_image_path)
            origin_image1 = cv2.resize(origin_image, (1024, 1024))
            white = np.array([255, 255, 255])   # 利用阈值算法将白色背景换成黑色背景（提取前景）
            threshold = 60  # 阈值可根据需要调整
            diff = np.abs(origin_image1 - white)
            diff_sum = np.sum(diff, axis=-1)
            mask = diff_sum < threshold
            origin_image1[mask] = [0, 0, 0]
            gray_origin_image = cv2.cvtColor(origin_image1, cv2.COLOR_BGR2GRAY)
            origin_image_HSV = cv2.cvtColor(origin_image1, cv2.COLOR_BGR2HSV)

            dna_image_gray = cv2.imread(dna_image_path, cv2.IMREAD_GRAYSCALE)
            dna_image_gray1 = cv2.resize(dna_image_gray, (1024, 1024))
            _, binary_image = cv2.threshold(dna_image_gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dna_image_color = cv2.bitwise_and(origin_image1, origin_image1, mask=binary_image)
            dna_image_HSV = cv2.cvtColor(dna_image_color, cv2.COLOR_BGR2HSV)
            gray_dna_image = cv2.cvtColor(dna_image_color, cv2.COLOR_BGR2GRAY)

            protein_image_gray = cv2.imread(protein_image_path, cv2.IMREAD_GRAYSCALE)
            protein_image_gray1 = cv2.resize(protein_image_gray, (1024, 1024))
            _, binary_image1 = cv2.threshold(protein_image_gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            protein_image_color = cv2.bitwise_and(origin_image1, origin_image1, mask=binary_image1)
            protein_image_HSV = cv2.cvtColor(protein_image_color, cv2.COLOR_BGR2HSV)
            gray_protein_image = cv2.cvtColor(protein_image_color, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("zzz1", origin_image1)
            # cv2.imshow("zzz2", dna_image_color)
            # cv2.imshow("zzz3", protein_image_HSV)
            # cv2.imshow("zzz4", dna_image_HSV)
            # cv2.imshow("zzz5", protein_image_HSV)
            # cv2.waitKey()
            # print(origin_image1.shape)
            # print(dna_image_color.shape)
            # print(protein_image_color.shape)
            B_origin = origin_image1[:, :, 0]
            G_origin = origin_image1[:, :, 1]
            R_origin = origin_image1[:, :, 2]
            H_origin = origin_image_HSV[:, :, 0]
            S_origin = origin_image_HSV[:, :, 1]
            V_origin = origin_image_HSV[:, :, 2]

            B_DNA = dna_image_color[:, :, 0]
            G_DNA = dna_image_color[:, :, 1]
            R_DNA = dna_image_color[:, :, 2]
            H_DNA = dna_image_HSV[:, :, 0]
            S_DNA = dna_image_HSV[:, :, 1]
            V_DNA = dna_image_HSV[:, :, 2]

            B_protein = protein_image_color[:, :, 0]
            G_protein = protein_image_color[:, :, 1]
            R_protein = protein_image_color[:, :, 2]
            H_protein = protein_image_HSV[:, :, 0]
            S_protein = protein_image_HSV[:, :, 1]
            V_protein = protein_image_HSV[:, :, 2]

            B_origin_mean = np.mean(B_origin[B_origin > 0])
            G_origin_mean = np.mean(G_origin[G_origin > 0])
            R_origin_mean = np.mean(R_origin[R_origin > 0])
            H_origin_mean = np.mean(H_origin[H_origin > 0])
            S_origin_mean = np.mean(S_origin[S_origin > 0])
            V_origin_mean = np.mean(V_origin[V_origin > 0])
            gray_origin_image_mean = np.mean(gray_origin_image[gray_origin_image > 0])
            gray_origin_image_ALL = np.sum(gray_origin_image)
            print(f"--{subfolder}--；--{dna_image}--；原图的Gray：{gray_origin_image_mean}---；B:{B_origin_mean}---；G：{G_origin_mean}---；R：{R_origin_mean}")

            B_DNA_mean = np.mean(B_DNA[B_DNA > 0])
            G_DNA_mean = np.mean(G_DNA[G_DNA > 0])
            R_DNA_mean = np.mean(R_DNA[R_DNA > 0])
            H_DNA_mean = np.mean(H_DNA[H_DNA > 0])
            S_DNA_mean = np.mean(S_DNA[S_DNA > 0])
            V_DNA_mean = np.mean(R_DNA[V_DNA > 0])
            gray_dna_image_mean = np.mean(gray_dna_image[gray_dna_image > 0])
            gray_dna_image_ALL = np.sum(gray_dna_image)

            B_protein_mean = np.mean(B_protein[B_protein > 0])
            G_protein_mean = np.mean(G_protein[G_protein > 0])
            R_protein_mean = np.mean(R_protein[R_protein > 0])
            H_protein_mean = np.mean(H_protein[H_protein > 0])
            S_protein_mean = np.mean(S_protein[S_protein > 0])
            V_protein_mean = np.mean(V_protein[V_protein > 0])
            gray_protein_image_mean = np.mean(gray_protein_image[gray_protein_image > 0])
            gray_protein_image_ALL = np.sum(gray_protein_image)

            traio_protein_DNA_meam =  gray_protein_image_mean / gray_dna_image_mean
            traio_protein_origin_meam = gray_protein_image_mean / gray_origin_image_mean
            traio_DNA_origin_meam = gray_dna_image_mean / gray_origin_image_mean

            traio_protein_DNA_ALL =  gray_protein_image_ALL / gray_dna_image_ALL
            traio_protein_origin_ALL = gray_protein_image_ALL / gray_origin_image_ALL
            traio_DNA_origin_ALL = gray_dna_image_ALL / gray_origin_image_ALL


            co_localization_ratios[(dna_image, subfolder)] = [B_origin_mean, G_origin_mean, R_origin_mean, gray_origin_image_mean, gray_origin_image_ALL, H_origin_mean, S_origin_mean, V_origin_mean,
                                                              B_DNA_mean, G_DNA_mean, R_DNA_mean, gray_dna_image_mean, gray_dna_image_ALL, H_DNA_mean, S_DNA_mean, V_DNA_mean,
                                                              B_protein_mean, G_protein_mean, R_protein_mean, gray_protein_image_mean, gray_protein_image_ALL, H_protein_mean, S_protein_mean, V_protein_mean,
                                                              traio_protein_DNA_meam, traio_protein_origin_meam, traio_DNA_origin_meam, traio_protein_DNA_ALL, traio_protein_origin_ALL, traio_DNA_origin_ALL]

    return co_localization_ratios

def pd_toexcel(data_dict, filename):
    # 创建一个空的 DataFrame
    df = pd.DataFrame()
    # 遍历字典中的每个键值对，按照您的格式将数据添加到 DataFrame 中
    for key, values in data_dict.items():
        image_name, subcellular_type = key
        df.at[len(df), '图像名称'] = image_name
        df.at[len(df)-1, '亚细胞种类'] = subcellular_type
        df.at[len(df)-1, 'B_origin_mean'] = values[0]
        df.at[len(df)-1, 'G_origin_mean'] = values[1]
        df.at[len(df)-1, 'R_origin_mean'] = values[2]
        df.at[len(df)-1, '原图像素均值'] = values[3]
        df.at[len(df)-1, '原图总像素值'] = values[4]
        df.at[len(df)-1, 'H_origin_mean'] = values[5]
        df.at[len(df)-1, 'S_origin_mean'] = values[6]
        df.at[len(df)-1, 'V_origin_mean'] = values[7]

        df.at[len(df)-1, 'B_DNA_mean'] = values[8]
        df.at[len(df)-1, 'G_DNA_mean'] = values[9]
        df.at[len(df)-1, 'R_DNA_mean'] = values[10]
        df.at[len(df)-1, 'DNA像素均值'] = values[11]
        df.at[len(df)-1, 'DNA总像素值'] = values[12]
        df.at[len(df)-1, 'H_DNA_mean'] = values[13]
        df.at[len(df)-1, 'S_DNA_mean'] = values[14]
        df.at[len(df)-1, 'V_DNA_mean'] = values[15]

        df.at[len(df)-1, 'B_protein_mean'] = values[16]
        df.at[len(df)-1, 'G_protein_mean'] = values[17]
        df.at[len(df)-1, 'R_protein_mean'] = values[18]
        df.at[len(df)-1, '蛋白质像素均值'] = values[19]
        df.at[len(df)-1, '蛋白质总像素值'] = values[20]
        df.at[len(df)-1, 'H_protein_mean'] = values[21]
        df.at[len(df)-1, 'S_protein_mean'] = values[22]
        df.at[len(df)-1, 'V_protein_mean'] = values[23]

        df.at[len(df)-1, '蛋白质与DNA灰度均值比值'] = values[24]
        df.at[len(df)-1, '蛋白质与原图灰度均值比值'] = values[25]
        df.at[len(df)-1, 'DNA与原图灰度均值比值'] = values[26]
        df.at[len(df)-1, '蛋白质与DNA总灰度值比值'] = values[27]
        df.at[len(df)-1, '蛋白质与原图总灰度值比值'] = values[28]
        df.at[len(df)-1, 'DNA与原图总灰度值比值'] = values[29]

    # 将 DataFrame 写入到 Excel 文件中
    df.to_excel(filename, index=False)

# 示例用法
dna_folder = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\Process_dataset\\DNA"
protein_folder = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\Process_dataset\\protein"
origin_folder = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\dataset"
subfolders = ['Cy', 'Er', 'Go', 'Mi', 'Nu', 'Ve']
co_localization_ratios = calculate_co_localization_ratio(dna_folder, protein_folder,origin_folder, subfolders)
pd_toexcel(co_localization_ratios, 'E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Feature_(RGBHSV_gray).xlsx')