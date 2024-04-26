import numpy as np
import torch.nn as nn
import torch
import warnings
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import os

import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------origin_start---------------------------------------------------
class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)
# ----------------------------------------------origin_end---------------------------------------------------
class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        self.conv1 = Conv(3, 64, 6, 2, 2)
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.C2f_1 = C2f(128, 128, n=3)
        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.C2f_2 = C2f(256, 256, n=6)
        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.C2f_3 = C2f(512, 512, n=9)
        self.conv5 = Conv(512, 1024, 3, 2, 1)
        self.C2f_4 = C2f(1024, 1024, n=3)
        self.sppf = SPPF(1024, 1024)

        # Neck
        self.conv_neck1 = Conv(1024, 512, 1, 1, 0)
        self.UP1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Concat()     512+512
        self.C2f_neck1 = C2f(1024, 512, 3)
        self.conv_neck2 = Conv(512, 256, 1,1, 0)
        self.UP2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Concat()     256+256
        self.C2f_neck2 = C2f(512, 256, 3)
        self.conv_neck3 = Conv(256, 256, 3,2, 1)
            # Concat()     256+256
        self.C2f_neck3 = C2f(512, 512, 3)
        self.conv_neck4 = Conv(512, 512, 3,2, 1)
            # Concat()     512+512
        self.C2f_neck4 = C2f(1024, 1024, 3)

        self.conv_Y1 = Conv(256, 1, 1, 1, 0)
        self.conv_Y2 = Conv(512, 1, 1, 1, 0)
        self.conv_Y3 = Conv(1024, 1, 1, 1, 0)

        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(10, 10))
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(10, 10))
        self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(10, 10))
        self.lin1 = nn.Sequential(
            nn.Linear(25600, 640),
            nn.ReLU(),
            nn.Dropout()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(51200, 320),
            nn.ReLU(),
            nn.Dropout()
        )
        self.lin3 = nn.Sequential(
            nn.Linear(102400, 160),
            nn.ReLU(),
            nn.Dropout()
        )
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)



    def forward(self, x):
        outputs = []
        Y_images = []
        Y_feature = []
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.C2f_1(x2)
        x4 = self.conv3(x3)
        x5 = self.C2f_2(x4)     # 第二个cat
        x6 = self.conv4(x5)
        x7 = self.C2f_3(x6)     # 第一个cat
        x8 = self.conv5(x7)
        x9 = self.C2f_4(x8)
        x10 = self.sppf(x9)

        x11 = self.conv_neck1(x10)   # 第四个cat
        x12 = self.UP1(x11)
        x13 = torch.cat([x7, x12], dim=1)
        x14 = self.C2f_neck1(x13)
        x15 = self.conv_neck2(x14)   # 第三个cat
        x16 = self.UP2(x15)
        x17 = torch.cat([x5, x16], dim=1)
        x18 = self.C2f_neck2(x17)                      # 输出的80x80x256的预测特征层
        x19 = self.conv_neck3(x18)
        x20 = torch.cat([x15, x19], dim=1)
        x21 = self.C2f_neck3(x20)                      # 输出的40x40x512的预测特征层
        x22 = self.conv_neck4(x21)
        x23 = torch.cat([x11, x22], dim=1)
        x24 = self.C2f_neck4(x23)                      # 输出的20x20x1024的预测特征层

        x18 = self.cbam1(x18)       # 是否启动在neck部分添加注意力机制
        x21 = self.cbam2(x21)       # 是否启动在neck部分添加注意力机制
        x24 = self.cbam3(x24)       # 是否启动在neck部分添加注意力机制


        # 将[1, 256, 80, 80]、[1, 512, 40, 40]和[1, 1024, 20, 20]这三个feature_map特征层怎么转化为一张可代表的图像，一共得到三张(取均值方式或者说用逐点卷积进行降维)
        y1 = self.conv_Y1(x18)
        y2 = self.conv_Y2(x21)
        y3 = self.conv_Y3(x24)


        # 将三个预测特征层利用全连检层和展平的方式将其转化为三种特征向量
        Y_feature_1 = self.avgpool1(x18)
        Y_feature_zz_1 = torch.flatten(Y_feature_1, 1)
        Y_feature_zzz_1 =  self.lin1(Y_feature_zz_1)

        Y_feature_2 = self.avgpool2(x21)
        Y_feature_zz_2 = torch.flatten(Y_feature_2, 1)
        Y_feature_zzz_2 =  self.lin2(Y_feature_zz_2)

        Y_feature_3 = self.avgpool3(x24)
        Y_feature_zz_3 = torch.flatten(Y_feature_3, 1)
        Y_feature_zzz_3 =  self.lin3(Y_feature_zz_3)


        outputs.append(x18)
        outputs.append(x21)
        outputs.append(x24)

        Y_feature.append(Y_feature_zzz_1)
        Y_feature.append(Y_feature_zzz_2)
        Y_feature.append(Y_feature_zzz_3)

        Y_images.append(y1)
        Y_images.append(y2)
        Y_images.append(y3)

        return outputs, Y_feature, Y_images


def process_image_and_outputs(model, image_path):
    # 读取图像并转换为张量
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 默认读取的图像格式是 BGR，需转换为 RGB
    image = cv2.resize(image, (640, 640))
    # cv2.imshow("zz", image)
    # print(image.shape)
    # cv2.waitKey()
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # 显式转换为浮点数类型

    # 将图像输入模型并获取输出特征层
    with torch.no_grad():
        outputs, Y_feature, Y_images = model(image_tensor)

    # 输出特征层的形状
    # output_shapes = [output.shape for output in outputs]
    return outputs, Y_feature, Y_images



Deep_feature_images = {}
m = 0
folder_path = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\data\\train_data\\dataset"
# folder_path = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\zz\\zzz"
feature_map_path1 = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map\\Image_80x80"
feature_map_path2 = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map\\Image_40x40"
feature_map_path3 = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map\\Image_20x20"

sub_folders = os.listdir(folder_path)
for sub_folder in sub_folders:
    sub_folder_path = os.path.join(folder_path, sub_folder)
    if not os.path.isdir(sub_folder_path):
        continue
    Images = os.listdir(sub_folder_path)
    for Image_file in Images:
        if Image_file.endswith(".jpg"):
            m += 1
            Image_file_path = os.path.join(sub_folder_path, Image_file)
            model = YOLOv8Backbone().to(device)
            outputs, Y_feature, Y_images = process_image_and_outputs(model, Image_file_path)

            Y1_feature_map = outputs[0].squeeze(0)
            Y2_feature_map = outputs[1].squeeze(0)
            Y3_feature_map = outputs[2].squeeze(0)

            Y1_feature_map_zz = torch.mean(Y1_feature_map, 0).cpu().numpy()
            Y2_feature_map_zz = torch.mean(Y2_feature_map, 0).cpu().numpy()
            Y3_feature_map_zz = torch.mean(Y3_feature_map, 0).cpu().numpy()

            Y1_feature_map_zzz = (Y1_feature_map_zz * 255).astype(np.uint8)
            Y2_feature_map_zzz = (Y2_feature_map_zz * 255).astype(np.uint8)
            Y3_feature_map_zzz = (Y3_feature_map_zz * 255).astype(np.uint8)


            # Y_feature_map_zz = [Y1_feature_map_zz, Y2_feature_map_zz, Y3_feature_map_zz]   用彩色进行可视化
            Y_feature_map_zz = [Y1_feature_map_zzz, Y2_feature_map_zzz, Y3_feature_map_zzz]
            fig = plt.figure(figsize=(30, 50))
            for i in range(len(Y_feature_map_zz)):
                a = fig.add_subplot(5, 4, i+1)
                imgplot = plt.imshow(Y_feature_map_zz[i])
                a.axis("off")
                # a.set_title(names[i].split('(')[0], fontsize = 30)
            plt.savefig(str(f'E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_map\\clolor\\feature_map_{m}.jpg'), bbox_inches='tight')
            plt.close()


            # mean_gray = Y1_feature_map_zzz.mean()
            # print(Y1_feature_map_zzz.shape)
            # print("灰度均值: {:.15f}".format(mean_gray))
            # cv2.imshow("zzz1", cv2.resize(Y1_feature_map_zzz, (1024, 1024)))
            # cv2.imshow("zzz2", cv2.resize(Y2_feature_map_zzz, (1024, 1024)))
            # cv2.imshow("zzz3", cv2.resize(Y3_feature_map_zzz, (1024, 1024)))
            # cv2.waitKey()

            # 用于feature_map图像的保存
            Image1 = Y_images[0].permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            Image2 = Y_images[1].permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            Image3 = Y_images[2].permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

            # Image1 = cv2.resize(Image1, (1024, 1024))
            # cv2.imshow("zz", Image1)
            # cv2.waitKey()

            Image1_path = os.path.join(feature_map_path1, sub_folder, Image_file)
            Image2_path = os.path.join(feature_map_path2, sub_folder, Image_file)
            Image3_path = os.path.join(feature_map_path3, sub_folder, Image_file)
            # cv2.imwrite(Image1_path, Image1)          # 用卷积进行降维得到的单一图像
            # cv2.imwrite(Image2_path, Image2)
            # cv2.imwrite(Image3_path, Image3)

            cv2.imwrite(Image1_path, Y1_feature_map_zzz)
            cv2.imwrite(Image2_path, Y2_feature_map_zzz)
            cv2.imwrite(Image3_path, Y3_feature_map_zzz)


            Y1_feature = Y_feature[0].detach().cpu().numpy()
            Y1_feature_zz = Y1_feature.squeeze(0)
            Y2_feature = Y_feature[1].detach().cpu().numpy()
            Y2_feature_zz = Y2_feature.squeeze(0)
            Y3_feature = Y_feature[2].detach().cpu().numpy()
            Y3_feature_zz = Y3_feature.squeeze(0)
            Y_feature_zz = np.concatenate([Y1_feature_zz, Y2_feature_zz, Y3_feature_zz])     # 文本特征的输出
            Deep_feature_images[(Image_file, sub_folder)] = Y_feature_zz
            print(f"图像名称:{Image_file}； 亚细胞种类：{sub_folder}；处理图像的数量：{m}")


def pd_toexcel(features_dict, filename):
    # 创建一个空的 DataFrame
    df = pd.DataFrame()
    # 遍历字典中的每个键值对，按照您的格式将数据添加到 DataFrame 中
    for key, values in features_dict.items():
        image_name, subcellular_type = key
        df.at[len(df), '图像名称'] = image_name
        df.at[len(df)-1, '亚细胞种类'] = subcellular_type
        for j in range(len(values)):
            df.at[len(df)-1, f'Deep{j+1}'] = values[j]
    # 将 DataFrame 写入到 Excel 文件中
    df.to_excel(filename, index=False)

output_excel = "E:\\Users\\zhangzhen\\Desktop\\Protein and subcellular segmentation\\Feature_outcome\\train_data\\Feature_(YOLOv8_ZZ).xlsx"
pd_toexcel(Deep_feature_images, output_excel)


# output_shapes = [output.shape for output in outputs]
# feature_shapes = [Y_Feature.shape for Y_Feature in Y_feature]
# image_shapes = [image.shape for image in Y_images]
# print("输出图像的形状：", Y1_feature_zz.shape, Y2_feature_zz.shape, Y3_feature_zz.shape, Y_feature_zz.shape)
# print("输出特征层的形状：", output_shapes)
# print("输出特征的形状：", feature_shapes)
# print("输出图像的形状：", image_shapes)

