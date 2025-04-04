import torch
import torch.nn as nn

pi = 3.141592653589793


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb



class Illumination_Estimator(nn.Module):
    """
    一个用于估计图像中的照明条件的神经网络模型。

    参数:
    - n_fea_middle: 中间特征层的特征数量。
    - n_fea_in: 输入特征的数量，默认为4。
    - n_fea_out: 输出特征的数量，默认为3。
    """

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        """
        初始化Illumination_Estimator网络结构。
        """
        super(Illumination_Estimator, self).__init__()

        # 第一个卷积层，用于将输入特征映射到中间特征空间。
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 深度可分离卷积层，用于在中间特征空间内部进行空间特征提取。
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        # 第二个卷积层，用于将中间特征映射到输出特征。
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        # print("Illumination_Estimator的三个卷积模块已经建立完成！")

        # time.sleep(2)

    def forward(self, img):
        """
        前向传播函数定义。

        参数:
        - img: 输入图像，形状为 (b, c=3, h, w)，其中 b 是批量大小, c 是颜色通道数, h 和 w 是图像的高度和宽度。

        返回:
        - illu_fea: 照明特征图。
        - illu_map: 输出的照明映射，形状为 (b, c=3, h, w)。
        """

        # 计算输入图像每个像素点在所有颜色通道上的平均值。
        mean_c = img.mean(dim=1).unsqueeze(1)  # 形状为 (b, 1, h, w) 对应公式中的Lp，也就是照明先验prior
        # print(f"照明先验的图片大小：{mean_c.shape}")

        # 将原始图像和其平均通道合并作为网络输入。
        input = torch.cat([img, mean_c], dim=1)  # 对应
        # print("原始图像和其平均通道合并图片大小:",input.shape)

        # 通过第一个卷积层处理。
        x_1 = self.conv1(input)

        # 应用深度可分离卷积提取特征。
        illu_fea = self.depth_conv(x_1)
        # print("照明特征图大小:",illu_fea.shape)

        # 通过第二个卷积层得到最终的照明映射。
        illu_map = self.conv2(illu_fea)
        # print("照明图片大小:",illu_map.shape)

        return illu_fea, illu_map