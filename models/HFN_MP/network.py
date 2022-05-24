import torch
import torch.nn as nn

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch.nn.init as init
from torchvision.transforms import Normalize

class PatternExtractor(torch.nn.Module):
    """
        Texture Transformer
    """
    def __init__(self, in_channels=3):
        super(PatternExtractor, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(2*in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(2 * in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Sigmoid()
        )



    def forward(self, x):
        meta_texture = self.encoder(x)
        reconstruct_rgb = self.decoder(meta_texture) # Useless

        return meta_texture, reconstruct_rgb

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        # return F.interpolate(x, size=(H,W), mode='bilinear') + y #
        return F.interpolate(x, size=(H, W), mode='nearest') + y  #

    def _downsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = x.size()
        # return F.interpolate(x, size=(H,W), mode='bilinear') + y #
        return F.interpolate(y, size=(H, W), mode='nearest') + x  #

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.latlayer1(c5)

        p4 = self._upsample_add(p5, self.latlayer2(c4))

        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        return p3, p7


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def FPN101():
    return FPN(Bottleneck, [2, 4, 23, 3])


class TwoStreamFPN(nn.Module):
    def __init__(self, FPN=FPN50):
        super(TwoStreamFPN, self).__init__()
        self.fpn_rgb = FPN()
        self.fpn_tex = FPN()


    def forward(self, rfb_input, tex_input):

        c1_rgb = F.relu(self.fpn_rgb.bn1(self.fpn_rgb.conv1(rfb_input)))
        c1_rgb = F.max_pool2d(c1_rgb, kernel_size=3, stride=2, padding=1)
        c2_rgb = self.fpn_rgb.layer1(c1_rgb)
        c3_rgb = self.fpn_rgb.layer2(c2_rgb)
        c4_rgb = self.fpn_rgb.layer3(c3_rgb)
        c5_rgb = self.fpn_rgb.layer4(c4_rgb)
        p6_rgb = self.fpn_rgb.conv6(c5_rgb)

        c1_tex = F.relu(self.fpn_tex.bn1(self.fpn_tex.conv1(tex_input)))
        c1_tex = F.max_pool2d(c1_tex, kernel_size=3, stride=2, padding=1)
        c2_tex = self.fpn_tex.layer1(c1_tex)
        c3_tex = self.fpn_tex.layer2(c2_tex)
        c4_tex = self.fpn_tex.layer3(c3_tex)
        c5_tex = self.fpn_tex.layer4(c4_tex)
        p6_tex = self.fpn_tex.conv6(c5_tex)

        p7_rgb = self.fpn_rgb.conv7(F.relu(p6_rgb))
        # Top-down
        p5_rgb = self.fpn_rgb.latlayer1(c5_rgb)
        p5_tex = self.fpn_tex.latlayer1(c5_tex)

        p5 = (p5_rgb + p5_tex) / 2
        c4 = (self.fpn_rgb.latlayer2(c4_rgb) + self.fpn_tex.latlayer2(c4_tex)) / 2
        p4_rgb = self.fpn_rgb._upsample_add(p5, c4)
        p4_tex = self.fpn_tex._upsample_add(p5, c4)

        p4_rgb = self.fpn_rgb.toplayer1(p4_rgb)
        p4_tex = self.fpn_tex.toplayer1(p4_tex)

        p4 = (p4_rgb + p4_tex) / 2

        c3 = (self.fpn_rgb.latlayer3(c3_rgb) + self.fpn_tex.latlayer3(c3_tex)) / 2

        p3_rgb = self.fpn_rgb._upsample_add(p4, c3)
        p3_tex = self.fpn_tex._upsample_add(p4, c3)
        p3_rgb = self.fpn_rgb.toplayer2(p3_rgb)
        p3_tex = self.fpn_tex.toplayer2(p3_tex)

        p3 = (p3_rgb + p3_tex) / 2

        p7_tex = self.fpn_tex.conv7(F.relu(p6_tex))
        # Top-down

        p7 = (p7_rgb + p7_tex) / 2

        return p3, p7


class HierachicalFusionNetwork(nn.Module):

    def __init__(self, mean_std_normalize=True, dropout_rate=0.0):
        super(HierachicalFusionNetwork, self).__init__()
        self.fpn = TwoStreamFPN()
        self.pred_head_0 = self._make_head()
        self.pred_head_cls_0 = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_heads = torch.nn.Linear(256, 2)
        self.mean_std_normalize = mean_std_normalize
        self.dropout = torch.nn.Dropout(dropout_rate)
        if mean_std_normalize:
            self.normalize = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.mean_std_normalize
    def forward(self, x_rgb, x_tex):
        if self.mean_std_normalize:
            normalized = []
            for i in range(x_tex.shape[0]):
                normalized.append(self.normalize(x_tex[i]))
            x_tex = torch.stack(normalized, 0)
        p3, p7 = self.fpn(x_rgb, x_tex)
        map_preds = []
        # pdb.set_trace()
        cls_preds = []

        map_preds.append(self.pred_head_0(p3))  # 32 -> 16 -> 8 -> 4 -> 2

        cls_out = self.pred_head_cls_0(p7)  # 32 -> 16 -> 8 -> 4 -> 2
        cls_out = cls_out.view(cls_out.size(0), -1)
        cls_out = self.dropout(cls_out)
        cls_out = self.cls_heads(cls_out)
        cls_preds.append(cls_out)

        return map_preds, cls_preds

    def _make_head(self):
        layers = []
        layers.append(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(128, 1, kernel_size=1))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def test():
    net = HierachicalFusionNetwork()
    loc_preds, cls_preds = net(Variable(torch.randn(2, 3, 256, 256)), Variable(torch.randn(2, 3, 256, 256)))
    print(loc_preds[0].size())
    print(cls_preds[0].size())


def get_state_dict():
    print('Loading pretrained ResNet50 model..')
    pretrained_model_path = 'models/resnet50_imagnet_pretrain.pth'
    save_model_path = 'models/HFN_MP/hfn_pretrain.pth'

    d = torch.load(pretrained_model_path)

    print('Loading into FPN50..')
    fpn = FPN50()
    dd = fpn.state_dict()
    for k in d.keys():
        if not k.startswith('fc'):  # skip fc layers
            dd[k] = d[k]

    print('Saving RetinaNet..')
    net = HierachicalFusionNetwork()
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # pi = 0.01
    # init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

    net.fpn.fpn_rgb.load_state_dict(dd)
    net.fpn.fpn_tex.load_state_dict(dd)
    torch.save(net.state_dict(), save_model_path)
    print('Done!')


if __name__ == '__main__':
    get_state_dict()
    test()
