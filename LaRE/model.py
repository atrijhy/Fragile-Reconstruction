import os
import types
import clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear
from torchvision.models.resnet import resnet50
from model_base import MultiHeadMapAttention, MultiHeadMapAttentionV2, ResNet, ResidualBlock

# 原代码使用作者本地的绝对路径加载 CLIP 权重，这在你机器上不存在。
# OpenAI-CLIP 支持直接用模型名称（如 "RN50", "RN50x64"）自动下载 / 调用缓存。
# 这里将映射改成模型名本身，兼容 clip.load(name, ...).
type_to_path = {
    'RN50x64': "RN50x64",
    'RN50': "RN50",
}


class Res50Feature(nn.Module):
    def __init__(self, num_class=4):
        super(Res50Feature, self).__init__()
        self.model = resnet50(pretrained=False)
        state_dict = torch.load(
            os.environ.get('RESNET50_CKPT', 'resnet50-19c8e357.pth'))
        self.model.load_state_dict(state_dict)
        print('load ok')
        self.model.fc = nn.Identity()

    def forward(self, image_input, isTrain=False):
        image_feats = self.model(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        return None, image_feats


class LASTED(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(LASTED, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.output_layer = Sequential(
        #     nn.Linear(1024, 1280),
        #     nn.GELU(),
        #     nn.Linear(1280, 512),
        # )
        # self.fc = nn.Linear(512, num_class)
        # self.text_input = clip.tokenize(['Real', 'Synthetic'])
        self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        # self.text_input = clip.tokenize(['Real-Photo', 'Synthetic-Photo', 'Real-Painting', 'Synthetic-Painting'])
        # self.text_input = clip.tokenize(['a', 'b', 'c', 'd'])

    def forward(self, image_input, isTrain=True):
        if isTrain:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats


class CLipClassifier(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifier, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)

    def forward(self, image_input):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        logits = self.fc(image_feats)
        return logits


class CLipClassifierV2(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifierV2, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)

    def forward(self, image_input, loss_map=''):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        logits = self.fc(image_feats)
        return logits


class CLipClassifierV3(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifierV3, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        self.clip_model.visual.attnpool = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, image_input, loss_map=''):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = self.pool(image_feats)
        image_feats = torch.flatten(image_feats, 1)
        logits = self.fc(image_feats)
        return logits


class CLipClassifierV4(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(CLipClassifierV4, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)

    def forward(self, image_input):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.clip_model.encode_image(image_input)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        logits = self.fc(image_feats)
        return logits


class ResnetBaseline(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50"):
        super(ResnetBaseline, self).__init__()
        self.model = resnet50()
        self.model.load_state_dict(torch.load(
            os.environ.get('RESNET50_CKPT', 'resnet50-19c8e357.pth')))
        self.model.fc = nn.Identity()
        # self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(2048, 2)

    def forward(self, image_input, loss_map):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        image_feats = self.model(image_input)  # bs * 2048
        logits = self.fc(image_feats)
        return logits


class CLipClassifierWMap(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMap, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)
        self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.attnpool = nn.Identity()

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        image_feats = self.clip_model.encode_image(image_input)
        # image_feats_attnpooled_normed = image_feats_attnpooled
        # print('init:', loss_map.shape)
        loss_map = self.conv(loss_map)  # bs * 1 * 32 * 32
        # print('conv:', loss_map.shape)
        loss_map_pooled = torch.mean(loss_map, dim=1, keepdim=True)  # bs * 1 * 32 * 32
        loss_map_pooled = F.adaptive_avg_pool2d(loss_map_pooled, (7, 7))
        spatial_weights = torch.sigmoid(loss_map_pooled)  # bs * 1 * 7 * 7

        feats_with_weights = image_feats * spatial_weights
        # feats_with_weights_pooled = F.adaptive_avg_pool2d(feats_with_weights, (1, 1)).squeeze() # bs * 2048
        image_feats = feats_with_weights
        image_feats_attnpooled = self.visual_attnpool(image_feats)

        final_feats = image_feats_attnpooled
        # print(image_feats.shape)
        # print(loss_map.shape)
        logits = self.fc(final_feats)
        return logits


class CLipClassifierWMapV2(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV2, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024, 2)
        self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.attnpool = nn.Identity()

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        image_feats = self.clip_model.encode_image(image_input)
        # image_feats_attnpooled_normed = image_feats_attnpooled
        # print('init:', loss_map.shape)
        loss_map = self.conv(loss_map)  # bs * 1 * 32 * 32
        # print('conv:', loss_map.shape)
        loss_map_pooled = torch.mean(loss_map, dim=1, keepdim=True)  # bs * 1 * 32 * 32
        loss_map_pooled = F.adaptive_avg_pool2d(loss_map_pooled, (7, 7))
        spatial_weights = torch.sigmoid(loss_map_pooled)  # bs * 1 * 7 * 7

        feats_with_weights = image_feats * spatial_weights
        image_feats_attnpooled = self.visual_attnpool(image_feats + feats_with_weights)
        logits = self.fc(image_feats_attnpooled)
        return logits


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class CLipClassifierWMapV3(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV3, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(512, 2)
        self.visual_attnpool = self.clip_model.visual.attnpool
        self.clip_model.visual.attnpool = nn.Identity()

        self.up_feature_map = Upsample(2048, 512, with_conv=True)
        self.down_loss_map = Downsample(4, 1, with_conv=True)

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats = self.clip_model.encode_image(image_input)  # bs * 2048 * 7 * 7
        up_image_feats = self.up_feature_map(image_feats)  # bs * 512 * 14 * 14

        down_loss_map = self.down_loss_map(loss_map)  # bs * 16 * 16 * 16
        down_loss_map = F.adaptive_avg_pool2d(down_loss_map, (14, 14))  # bs * 1 * 14 * 14

        up_image_feats = up_image_feats * torch.sigmoid(down_loss_map)
        global_feats = F.adaptive_avg_pool2d(up_image_feats, (1, 1)).squeeze()  # bs * 512

        logits = self.fc(global_feats)
        return logits


class CLIPVisualWrapper(nn.Module):
    """
    Wrapper for CLIP visual module that returns both final features and layer3 features.
    This is a proper nn.Module so it works with DataParallel.
    """
    def __init__(self, clip_visual):
        super().__init__()
        # Store all the CLIP visual components as proper submodules
        self.conv1 = clip_visual.conv1
        self.bn1 = clip_visual.bn1
        self.relu1 = clip_visual.relu1
        self.conv2 = clip_visual.conv2
        self.bn2 = clip_visual.bn2
        self.relu2 = clip_visual.relu2
        self.conv3 = clip_visual.conv3
        self.bn3 = clip_visual.bn3
        self.relu3 = clip_visual.relu3
        self.avgpool = clip_visual.avgpool
        self.layer1 = clip_visual.layer1
        self.layer2 = clip_visual.layer2
        self.layer3 = clip_visual.layer3
        self.layer4 = clip_visual.layer4
        self.attnpool = clip_visual.attnpool

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        # Stem
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x_l3 = self.layer3(x)
        x = self.layer4(x_l3)
        x = self.attnpool(x)
        return x, x_l3


class CLIPModelWrapper(nn.Module):
    """
    Wrapper for CLIP model that uses the modified visual module.
    This is a proper nn.Module so it works with DataParallel.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = CLIPVisualWrapper(clip_model.visual)
        self.dtype = clip_model.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))


class CLipClassifierWMapV4(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV4, self).__init__()
        clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # Wrap CLIP model with proper nn.Module wrapper for DataParallel compatibility
        self.clip_model = CLIPModelWrapper(clip_model)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024, 2)

        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(image_input)
        # block3 bs * 1024 * 14 * 14
        # print(block3_feats.shape)
        # up_image_feats = self.up_feature_map(image_feats)  # bs * 1024 * 14 * 14

        loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(loss_map)  # bs * 1 * 14 * 14
        up_image_feats = block3_feats * torch.sigmoid(pooled_loss_map)
        global_feats = F.adaptive_avg_pool2d(up_image_feats, (1, 1)).squeeze()  # bs * 512

        logits = self.fc(torch.cat([image_feats, global_feats], dim=1))
        return logits


class CLipClassifierWMapV5(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV5, self).__init__()
        clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # Wrap CLIP model with proper nn.Module wrapper for DataParallel compatibility
        self.clip_model = CLIPModelWrapper(clip_model)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024, 2)

        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.attn_pool = MultiHeadMapAttention()

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(image_input)
        # block3 bs * 1024 * 14 * 14
        # print(block3_feats.shape)
        # up_image_feats = self.up_feature_map(image_feats)  # bs * 1024 * 14 * 14

        loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(loss_map)  # bs * 8 * 14 * 14

        pooled_block3_feats = self.attn_pool(block3_feats, pooled_loss_map)  # bs * 1024
        # print(pooled_block3_feats.shape)
        # up_image_feats = block3_feats * torch.sigmoid(pooled_loss_map)
        # global_feats = F.adaptive_avg_pool2d(up_image_feats, (1, 1)).squeeze()  # bs * 512
        logits = self.fc(torch.cat([image_feats, pooled_block3_feats], dim=1))
        return logits


class CLipClassifierWMapV6(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV6, self).__init__()
        clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # Wrap CLIP model with proper nn.Module wrapper for DataParallel compatibility
        self.clip_model = CLIPModelWrapper(clip_model)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024 + 1024, 2)

        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.conv_align = nn.Conv2d(4, 1024, kernel_size=(1, 1))
        self.attn_pool = MultiHeadMapAttention()
        self.channel_align = ChannelAlignLayer(4, 128, 1024)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(image_input)
        # block3 bs * 1024 * 14 * 14
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(aligned_loss_map)  # bs * 8 * 14 * 14
        pooled_block3_feats = self.attn_pool(block3_feats, pooled_loss_map)  # bs * 1024

        channel_weighted_feats = self.channel_align(block3_feats, loss_map)  # bs * 1024
        logits = self.fc(torch.cat([image_feats, pooled_block3_feats, channel_weighted_feats], dim=1))
        return logits


class CLipClassifierOnlyMapV1(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierOnlyMapV1, self).__init__()
        # self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.model = ResNet(ResidualBlock, [2, 2, 2], 2)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        logits = self.model(loss_map)
        return logits


class CLipClassifierOnlyMapV2(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierOnlyMapV2, self).__init__()
        # self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.model = ResNet(ResidualBlock, [2, 2, 2], 2)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        logits = self.model(loss_map)
        return logits


class CLipClassifierWMapV7(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapV7, self).__init__()
        clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # Wrap CLIP model with proper nn.Module wrapper for DataParallel compatibility
        self.clip_model = CLIPModelWrapper(clip_model)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024 + 1024 + 1024, 2)

        self.conv = nn.Conv2d(4, 1024, kernel_size=(1, 1))  # for 8 heads
        self.conv_align = nn.Conv2d(4, 1024, kernel_size=(1, 1))
        # self.attn_pool = MultiHeadMapAttention()
        self.channel_align = ChannelAlignLayer(4, 128, 1024)
        self.attn_pool = MultiHeadMapAttentionV2()

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(image_input)
        # block3 bs * 1024 * 14 * 14
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        upscaled_loss_map = self.conv(aligned_loss_map)  # bs * 1024 * 14 * 14

        refined_feats = self.attn_pool(block3_feats, upscaled_loss_map)  # bs * 2048

        channel_weighted_feats = self.channel_align(block3_feats, loss_map)  # bs * 1024
        logits = self.fc(torch.cat([image_feats, refined_feats, channel_weighted_feats], dim=1))
        return logits


class PoolWithMap(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, loss_map):
        # x bs * 1024 * 14 * 14
        # loss_map bs * 1024 * 14 * 14
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        loss_map = loss_map.flatten(start_dim=2).permute(2, 0, 1)
        loss_map = torch.cat([loss_map.mean(dim=0, keepdim=True), loss_map], dim=0)
        loss_map = loss_map + self.positional_embedding[:, None, :].to(loss_map.dtype)  # (HW+1)NC

        queries = torch.cat([x[:1], loss_map[:1]], dim=0)  # 2 * N * C

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ChannelAlignLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ChannelAlignLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_map, loss_map):
        b, c, _, _ = feature_map.size()
        # 保持 batch 维度，避免在 bs=1 时 squeeze 掉导致维度不一致
        pooled_feature_map = self.avg_pool(feature_map).view(b, -1)  # [B,C]
        pooled_loss_map = self.avg_pool(loss_map).view(b, -1)  # [B,in_dim]
        weights = self.fc(pooled_loss_map)
        out = pooled_feature_map * weights
        return out


class CLipClassifierWMapBaseline(nn.Module):
    def __init__(self, num_class=4, clip_type="RN50x64"):
        super(CLipClassifierWMapBaseline, self).__init__()
        self.clip_model, self.preprocess = clip.load(type_to_path[clip_type], device='cpu', jit=False)
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(2048, 2)
        self.clip_model.visual.attnpool = nn.Identity()
        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1))

    def forward(self, image_input, loss_map):
        image_feats = self.clip_model.encode_image(image_input)
        loss_map = self.conv(loss_map)  # bs * 1 * 32 * 32
        loss_map_pooled = torch.mean(loss_map, dim=1, keepdim=True)  # bs * 1 * 32 * 32
        loss_map_pooled = F.adaptive_avg_pool2d(loss_map_pooled, (7, 7))
        spatial_weights = torch.sigmoid(loss_map_pooled)  # bs * 1 * 7 * 7

        feats_with_weights = image_feats * spatial_weights
        final_feats = F.adaptive_avg_pool2d(feats_with_weights, (1, 1)).squeeze()
        logits = self.fc(final_feats)
        return logits


class LASTEDWLoss(nn.Module):
    def __init__(self, num_class=4):
        super(LASTEDWLoss, self).__init__()
        self.clip_model, self.preprocess = clip.load(
            os.environ.get('CLIP_RN50x64_PATH', 'RN50x64.pt'), device='cpu', jit=False)
        # self.output_layer = Sequential(
        #     nn.Linear(1024, 1280),
        #     nn.GELU(),
        #     nn.Linear(1280, 512),
        # )
        # self.fc = nn.Linear(512, num_class)
        # self.text_input = clip.tokenize(['Real', 'Synthetic'])
        self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        # self.text_input = clip.tokenize(['Real-Photo', 'Synthetic-Photo', 'Real-Painting', 'Synthetic-Painting'])
        # self.text_input = clip.tokenize(['a', 'b', 'c', 'd'])

    def forward(self, image_input, isTrain=True):
        if isTrain:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = LASTED().cuda()
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params: %.2f' % (params / (1024 ** 2)))

    x = torch.zeros([4, 3, 448, 448]).cuda()
    _, logits = model(x)
    print(logits.shape)
