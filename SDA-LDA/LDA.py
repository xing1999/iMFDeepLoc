#coding:utf-8
import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''


def lda(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''

    clusters = np.unique(target)

    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    #within_class scatter matrix
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    #between_class scatter matrix
    SB = np.zeros((data.shape[1],data.shape[1]))
    u = data.mean(0)  #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  #某个类别的平均值
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi
    S = np.linalg.inv(Sw)*SB
    eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim-1):-1]
    w = eigVects[:,eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim


def process_lda(input_folder, output_folder):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 处理每个文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.xlsx'):
            # 读取数据
            df = pd.read_excel(os.path.join(input_folder, file_name), header=None, engine='openpyxl')
            last_column = df.iloc[:, -1]
            last_column_matrix = last_column.values
            Y1 = last_column_matrix[1:]
            label_encoder = LabelEncoder()
            Y1_encoded = label_encoder.fit_transform(Y1)

            X1 = df.iloc[1:, :-1]

            # LDA
            data_2 = LinearDiscriminantAnalysis(n_components=5).fit_transform(X1, Y1_encoded)

            # 合并数据
            merged_array = np.column_stack((data_2, Y1))

            # 创建 DataFrame
            df_output = pd.DataFrame(merged_array)
            remarks_row = ['LDA' + str(i + 1) for i in range(merged_array.shape[1] - 1)] + ['亚细胞种类']
            df_output = pd.concat([pd.DataFrame([remarks_row], columns=df_output.columns), df_output], ignore_index=True)

            # 写入 Excel 文件
            output_file_name = f"LDA_{os.path.splitext(file_name)[0]}.xlsx"
            output_path = os.path.join(output_folder, output_file_name)
            df_output.to_excel(output_path, index=False, header=False, startrow=0)
            print(f"文件 '{output_file_name}' 已成功写入！")

# 指定输入文件夹和输出文件夹路径
input_folder = 'SDA_Normalized_label_datasets'
output_folder = 'LDA_SDA_Normalized_label_datasets'

# 执行处理
process_lda(input_folder, output_folder)





# if __name__ == '__main__':
#     iris = load_iris()
#     X = iris.data
#     Y = iris.target
#     df = pd.read_excel('E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Label_datasets_TXT\\(1)_Feature_(RGBHSV_gray_30).xlsx', header=None, engine='openpyxl')
#     last_column = df.iloc[:, -1]
#     last_column_matrix = last_column.values
#     Y1 = last_column_matrix[1:]
#     label_encoder = LabelEncoder()
#     Y1_encoded = label_encoder.fit_transform(Y1)
#
#     X1 = df.iloc[1:, :-1]
#
#     # data_1 = lda(X1, Y1, 4)
#     data_2 = LinearDiscriminantAnalysis(n_components=5).fit_transform(X1, Y1_encoded)
#     merged_array = np.column_stack((data_2, Y1))
#
#     df = pd.DataFrame(merged_array)
#     remarks_row = ['LDA' + str(i + 1) for i in range(merged_array.shape[1] - 1)] + ['亚细胞种类']
#     df = pd.concat([pd.DataFrame([remarks_row], columns=df.columns), df], ignore_index=True)
#     df.to_excel('LDA.xlsx', index=False, header=False, startrow=0)
#
#
#
#     print("Excel 文件已成功写入！")
    # plt.figure(figsize=(8,4))
    # plt.subplot(121)
    # plt.title("my_LDA")
    # plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    # plt.subplot(122)
    # plt.title("sklearn_LDA")
    # plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    # plt.savefig("LDA.png")
    # plt.show()