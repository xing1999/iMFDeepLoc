import cv2
import numpy as np
import os
import pandas as pd
from pylab import*

# ---------------------------------------LBP系列的运算代码---------------------------------------------
class LBP:
    def __init__(self):
        # revolve_map为旋转不变模式的36种特征值从小到大进行序列化编号得到的字典
        self.revolve_map = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8, 17: 9, 19: 10, 21: 11, 23: 12,
                            25: 13, 27: 14, 29: 15, 31: 16, 37: 17, 39: 18, 43: 19, 45: 20, 47: 21, 51: 22, 53: 23,
                            55: 24,
                            59: 25, 61: 26, 63: 27, 85: 28, 87: 29, 91: 30, 95: 31, 111: 32, 119: 33, 127: 34, 255: 35}
        # uniform_map为等价模式的58种特征值从小到大进行序列化编号得到的字典
        self.uniform_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8,
                            14: 9, 15: 10, 16: 11, 24: 12, 28: 13, 30: 14, 31: 15, 32: 16,
                            48: 17, 56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23, 112: 24,
                            120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32,
                            143: 33, 159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40,
                            223: 41, 224: 42, 225: 43, 227: 44, 231: 45, 239: 46, 240: 47, 241: 48,
                            243: 49, 247: 50, 248: 51, 249: 52, 251: 53, 252: 54, 253: 55, 254: 56,
                            255: 57}

    # 图像的LBP原始特征计算算法：将图像指定位置的像素与周围8个像素比较
    # 比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calute_basic_lbp(self, image_array, i, j):
        sum = []
        if image_array[i - 1, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i - 1, j] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i - 1, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum

    # 获取二进制序列进行不断环形旋转得到新的二进制序列的最小十进制值
    def get_min_for_revolve(self, arr):
        values = []
        circle = arr
        circle.extend(arr)
        for i in range(0, 8):
            j = 0
            sum = 0
            bit_num = 0
            while j < 8:
                sum += circle[i + j] << bit_num
                bit_num += 1
                j += 1
            values.append(sum)
        return min(values)

    # 获取值r的二进制中1的位数
    def calc_sum(self, r):
        num = 0
        while (r):
            r &= (r - 1)
            num += 1
        return num

    # 获取图像的LBP原始模式特征
    def lbp_basic(self, image_array):
        basic_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                sum = self.calute_basic_lbp(image_array, i, j)
                bit_num = 0
                result = 0
                for s in sum:
                    result += s << bit_num
                    bit_num += 1
                basic_array[i, j] = result
        return basic_array

    # 获取图像的LBP旋转不变模式特征
    def lbp_revolve(self, image_array):
        revolve_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                sum = self.calute_basic_lbp(image_array, i, j)
                revolve_key = self.get_min_for_revolve(sum)
                revolve_array[i, j] = self.revolve_map[revolve_key]
        return revolve_array

    # 获取图像的LBP等价模式特征
    def lbp_uniform(self, image_array):
        uniform_array = np.zeros(image_array.shape, np.uint8)
        basic_array = self.lbp_basic(image_array)
        width = image_array.shape[0]
        height = image_array.shape[1]

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                k = basic_array[i, j] << 1
                if k > 255:
                    k = k - 255
                xor = basic_array[i, j] ^ k
                num = self.calc_sum(xor)
                if num <= 2:
                    uniform_array[i, j] = self.uniform_map[basic_array[i, j]]
                else:
                    uniform_array[i, j] = 58
        return uniform_array

    # 获取图像的LBP旋转不变等价模式特征
    def lbp_revolve_uniform(self, image_array):
        uniform_revolve_array = np.zeros(image_array.shape, np.uint8)
        basic_array = self.lbp_basic(image_array)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                k = basic_array[i, j] << 1
                if k > 255:
                    k = k - 255
                xor = basic_array[i, j] ^ k
                num = self.calc_sum(xor)
                if num <= 2:
                    uniform_revolve_array[i, j] = self.calc_sum(basic_array[i, j])
                else:
                    uniform_revolve_array[i, j] = 9
        return uniform_revolve_array

    # 绘制指定维数和范围的图像灰度归一化统计直方图
    def show_hist(self, img_array, im_bins, im_range):
        hist = cv2.calcHist([img_array], [0], None, im_bins, im_range)
        hist = cv2.normalize(hist, None).flatten()
        plt.plot(hist, color='r')
        plt.xlim(im_range)
        plt.show()

    # 绘制图像原始LBP特征的归一化统计直方图
    def show_basic_hist(self, img_array):
        self.show_hist(img_array, [256], [0, 256])

    # 绘制图像旋转不变LBP特征的归一化统计直方图
    def show_revolve_hist(self, img_array):
        self.show_hist(img_array, [36], [0, 36])

    # 绘制图像等价模式LBP特征的归一化统计直方图
    def show_uniform_hist(self, img_array):
        self.show_hist(img_array, [60], [0, 60])

    # 绘制图像旋转不变等价模式LBP特征的归一化统计直方图
    def show_revolve_uniform_hist(self, img_array):
        self.show_hist(img_array, [10], [0, 10])

    # 显示图像
    def show_image(self, image_array):
        cv2.imshow('Image', image_array)
        cv2.waitKey(0)
# ---------------------------------------LBP系列的运算代码---------------------------------------------

def Image_preprocessing(image_path):
    protein_image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(protein_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    protein_image_gray1 = cv2.bitwise_and(protein_image_gray, protein_image_gray, mask=binary_image)
    protein_image_gray2 = cv2.resize(protein_image_gray1, (1024, 1024))

    return protein_image_gray2


def get_local_region_lbph(src, min_value, max_value, normed=True):
    hist_size = max_value - min_value + 1
    hist_range = (min_value, max_value + 1)
    hist = cv2.calcHist([src], [0], None, [hist_size], hist_range, accumulate=False)

    if normed:
        hist /= src.size

    return hist.reshape(1, -1)

def get_lbph(src, num_patterns, grid_x, grid_y, normed=True):
    width = src.shape[1] // grid_x
    height = src.shape[0] // grid_y
    result = np.zeros((grid_x * grid_y, num_patterns), dtype=np.float32)

    result_row_index = 0
    for i in range(grid_x):
        for j in range(grid_y):
            src_cell = src[i*height : (i+1)*height, j*width : (j+1)*width]
            hist_cell = get_local_region_lbph(src_cell, 0, num_patterns - 1, normed)
            result[result_row_index] = hist_cell
            result_row_index += 1

    return result.reshape(1, -1)


def extract_features(image_folder, num_patterns, grid_x, grid_y, normed=True):
    lbp = LBP()
    features_dict = {}
    for sub_folder in os.listdir(image_folder):
        sub_folder_path = os.path.join(image_folder, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue
        for filename in os.listdir(sub_folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(sub_folder_path, filename)
                gray_image = Image_preprocessing(image_path)
                # LBP_image = lbp.lbp_basic(gray_image)   # zz lbp_uniform
                # LBP_image = lbp.lbp_revolve(gray_image)   # zz
                LBP_image = lbp.lbp_uniform(gray_image)  # zz
                # LBP_image = lbp.lbp_revolve_uniform(gray_image)  # zz
                lbph_features = get_lbph(LBP_image, num_patterns, grid_x, grid_y, normed)
                # print(f"图像名称:{filename}； 亚细胞种类：{sub_folder}；lbph_features：{lbph_features}")
                print(f"图像名称:{filename}； 亚细胞种类：{sub_folder}")
                features_dict[(filename, sub_folder)] = lbph_features.flatten()

            def pd_toexcel(features_dict, filename):
                # 创建一个空的 DataFrame
                df = pd.DataFrame()
                # 遍历字典中的每个键值对，按照您的格式将数据添加到 DataFrame 中
                for key, values in features_dict.items():
                    image_name, subcellular_type = key
                    df.at[len(df), '图像名称'] = image_name
                    df.at[len(df) - 1, '亚细胞种类'] = subcellular_type
                    for j in range(len(values)):
                        df.at[len(df) - 1, f'LBP{j + 1}'] = values[j]
                # 将 DataFrame 写入到 Excel 文件中
                df.to_excel(filename, index=False)
            pd_toexcel(features_dict, output_excel)

    return features_dict



if __name__ == '__main__':
    num_patterns = 58  # LBP值的模式种类

    grid_x = 1
    grid_y = 1
    normed = True
    image_folder = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\Process_dataset\\LBP_uniform1"
    output_excel = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Feature_(LBP_uniform).xlsx"
    features_dict = extract_features(image_folder, num_patterns, grid_x, grid_y, normed)





