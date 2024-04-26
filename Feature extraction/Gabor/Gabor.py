import cv2
import os
import glob
import numpy as np
import pandas as pd
import time

tic = time.time()

# Main directory where all the images are stored
main_dir = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map2/Image_20x20"

# List of subfolders
subfolders = ['Cy', 'Er', 'Go', 'Mi', 'Nu', 'Ve']
# subfolders = ['afq']


gamma = 0.5
sigma = 0.56
theta_list = [0, np.pi, np.pi / 2, np.pi / 4, 3 * np.pi / 4]  # Angles
phi = 0
lamda_list = [2 * np.pi / 1, 2 * np.pi / 2, 2 * np.pi / 3, 2 * np.pi / 4, 2 * np.pi / 5]  # Wavelengths
num = 1

# Creating headings for the DataFrame
gabor_label = ['Image', 'Subfolder']
for i in range(50):
    gabor_label.append('Gabor' + str(i + 1))

data = []

for subfolder in subfolders:
    subfolder_path = os.path.join(main_dir, subfolder)

    # Iterate through all images in the subfolder
    for img_file in os.listdir(subfolder_path):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(subfolder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            print("Processing image: {}".format(img_path))

            local_energy_list = []
            mean_ampl_list = []

            for theta in theta_list:
                print("For theta = {:.2f}pi".format(theta / np.pi))

                for lamda in lamda_list:
                    kernel = cv2.getGaborKernel((3, 3), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                    fimage = cv2.filter2D(img, cv2.CV_8UC3, kernel)

                    mean_ampl = np.sum(np.abs(fimage))
                    mean_ampl_list.append(mean_ampl)

                    local_energy = np.sum(fimage ** 2)
                    local_energy_list.append(local_energy)

                    num += 1

            # Append the features along with image and subfolder information
            row_data = [img_file, subfolder] + local_energy_list + mean_ampl_list
            data.append(row_data)

# Create a DataFrame
df = pd.DataFrame(data, columns=gabor_label)

# Write DataFrame to Excel file
excel_file = 'Feature_(Gabor_50+).xlsx'
df.to_excel(excel_file, index=False)

toc = time.time()
print("Computation time is {:.2f} seconds".format(toc - tic))
