import os
import pandas as pd

# 文件夹路径
folder_path = 'zz/excel'

# 用于存储所有数据的DataFrame
combined_df = pd.DataFrame()

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        # 读取Excel文件
        file_path = os.path.join(folder_path, filename)
        df = pd.read_excel(file_path, engine='openpyxl')


        # 提取第三列及之后的数据
        df_columns = df.columns.tolist()[0:]  # 从第三列开始
        df_subset = df[df_columns]

        # 将数据添加到总的DataFrame中
        combined_df = pd.concat([combined_df, df_subset], axis=1)

# 将合并后的DataFrame写入新的Excel文件
combined_df.to_excel('zz/excel/(tradition+2+11+12+13)_Feature.xlsx', index=False)
