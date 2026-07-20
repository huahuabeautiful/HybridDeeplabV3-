import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

# 在文件顶部的 import 区域添加：
try:
    from .axial_attention import AxialAttention
except ImportError:
    from nets.axial_attention import AxialAttention

# 检查可变形卷积可用性
try:
    from torchvision.ops import DeformConv2d

    deform_conv_available = True
except ImportError:
    deform_conv_available = False
    import warnings

    warnings.warn("DeformConv2d not available, using standard convolution instead")


class HybridASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, deformable_groups=4):
        super(HybridASPP, self).__init__()

        # 1. 标准1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 2. 深度可分离空洞卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, 1, padding=6 * rate, dilation=6 * rate,
                      groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 3. 标准空洞卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 4. 可变形卷积分支（或备用标准卷积）
        if deform_conv_available:
            # 使用正确的偏移量通道数 (2 * kernel_size * kernel_size * deformable_groups)
            self.offset_conv = nn.Conv2d(
                dim_in,
                2 * 3 * 3 * deformable_groups,  # 2 * kernel_size[0] * kernel_size[1] * deformable_groups
                3, 1, padding=1, bias=True
            )

            # 计算正确的膨胀率和padding
            dilation_val = max(1, 18 * rate // 12)
            padding_val = dilation_val  # 对于kernel_size=3，padding应等于dilation

            # 检测PyTorch版本以确定参数名
            torch_version = torch.__version__.split('.')
            major_version = int(torch_version[0]) if torch_version[0].isdigit() else 0
            minor_version = int(torch_version[1]) if len(torch_version) > 1 and torch_version[1].isdigit() else 0

            # 对于PyTorch 2.0及以上版本使用新API
            if major_version >= 2:
                self.deform_conv = DeformConv2d(
                    dim_in, dim_out, kernel_size=3, stride=1, padding=padding_val,
                    dilation=dilation_val, bias=True
                )
            else:
                # 对于旧版本使用旧API
                self.deform_conv = DeformConv2d(
                    dim_in, dim_out, kernel_size=3, stride=1, padding=padding_val,
                    dilation=dilation_val, bias=True,
                    deformable_groups=deformable_groups
                )

            self.branch4_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
            self.branch4_relu = nn.ReLU(inplace=True)
            self.use_deform_conv = True
        else:
            # 对于标准卷积分支，使用相同的dilation和padding
            dilation_val = max(1, 18 * rate // 12)
            padding_val = dilation_val
            self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=padding_val,
                          dilation=dilation_val, bias=True),
                nn.BatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            self.use_deform_conv = False

        # 5. 全局上下文分支
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 特征融合（移除了最后的注意力机制）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out * 2, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out * 2, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out * 2, dim_out, 3, 1, padding=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 各分支前向传播
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        # 分支4处理
        if self.use_deform_conv:
            offset = self.offset_conv(x)
            branch4 = self.deform_conv(x, offset)
            branch4 = self.branch4_bn(branch4)
            branch4 = self.branch4_relu(branch4)
        else:
            branch4 = self.branch4(x)

        # 全局上下文分支
        global_feat = torch.mean(x, [2, 3], keepdim=True)
        global_feat = self.branch5_conv(global_feat)
        global_feat = self.branch5_bn(global_feat)
        global_feat = self.branch5_relu(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=True)

        # 特征融合（移除了注意力加权）
        fused = torch.cat([branch1, branch2, branch3, branch4, global_feat], dim=1)
        output = self.fusion_conv(fused)  # 直接使用融合结果作为输出

        return output


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class DeepLab(nn.Module):
    # ========================================================== #
    #   修改：加入 use_axial_attention 参数控制是否使用轴向注意力
    # ========================================================== #
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16,
                 use_axial_attention=False):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   替换为HybridASPP特征提取模块
        #   利用混合卷积进行多尺度特征提取
        # -----------------------------------------#
        self.aspp = HybridASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        # ========================================== #
        #   新增：实例化轴向注意力机制
        # ========================================== #
        self.use_axial_attention = use_axial_attention
        if self.use_axial_attention:
            self.axial_attention = AxialAttention(in_channels=256)

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        # ========================================== #
        #   新增：在此处施加轴向注意力机制
        # ========================================== #
        if self.use_axial_attention:
            x = self.axial_attention(x)

        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x