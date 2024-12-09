from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d
import torch
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.transforms import Normalize

class MetaPatternExtractor(MetaModule):
    """
        Texture Transformer compatible with MetaModule
    """
    def __init__(self, in_channels=3):
        super(MetaPatternExtractor, self).__init__()

        self.encoder = MetaSequential(
            MetaConv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(2 * in_channels),
            torch.nn.ReLU(),
            MetaConv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(in_channels),
            torch.nn.Sigmoid()
        )

        self.decoder = MetaSequential(
            MetaConv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(2 * in_channels),
            torch.nn.ReLU(),
            MetaConv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(in_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x, params=None):
        if params is None:
            meta_texture = self.encoder(x)
            reconstruct_rgb = self.decoder(meta_texture)
        else:
            encoder_params = params.get('encoder', None)
            decoder_params = params.get('decoder', None)
            
            meta_texture = self.encoder(x, params=encoder_params)
            reconstruct_rgb = self.decoder(meta_texture, params=decoder_params)
            
        return meta_texture, reconstruct_rgb
    
class MetaPatternExtractorWithSkip(MetaModule):
    def __init__(self, in_channels=3):
        super(MetaPatternExtractorWithSkip, self).__init__()
        self.encoder = MetaSequential(
            MetaConv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(2 * in_channels),
            torch.nn.ReLU(),
            MetaConv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(in_channels),
            torch.nn.Sigmoid()
        )

        self.decoder = MetaSequential(
            MetaConv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(2 * in_channels),
            torch.nn.ReLU(),
            MetaConv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            MetaBatchNorm2d(in_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x, params=None):
        if params is None:
            meta_texture = self.encoder(x)
            reconstruct_rgb = self.decoder(meta_texture + x)  # Skip Connection
        else:
            encoder_params = params.get('encoder', None)
            decoder_params = params.get('decoder', None)
            
            meta_texture = self.encoder(x, params=encoder_params)
            reconstruct_rgb = self.decoder(meta_texture + x, params=decoder_params)  # Skip Connection
            
        return meta_texture, reconstruct_rgb
    


class MetaBottleneck(MetaModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(MetaBottleneck, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(self.expansion * planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = MetaSequential(
                MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(self.expansion * planes)
            )
        else:
            self.downsample = MetaSequential()

    def forward(self, x, params=None):
        out = F.relu(self.bn1(self.conv1(x, params=self.get_subdict(params, 'conv1')), 
                             params=self.get_subdict(params, 'bn1')))
        out = F.relu(self.bn2(self.conv2(out, params=self.get_subdict(params, 'conv2')), 
                             params=self.get_subdict(params, 'bn2')))
        out = self.bn3(self.conv3(out, params=self.get_subdict(params, 'conv3')), 
                      params=self.get_subdict(params, 'bn3'))
        out += self.downsample(x, params=self.get_subdict(params, 'downsample'))
        out = F.relu(out)
        return out

class MetaFPN(MetaModule):
    def __init__(self, block, num_blocks):
        super(MetaFPN, self).__init__()
        self.in_planes = 64

        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = MetaBatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.conv6 = MetaConv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = MetaConv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.latlayer1 = MetaConv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = MetaConv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = MetaConv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.toplayer1 = MetaConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = MetaConv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return MetaSequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def _downsample_add(self, x, y):
        _, _, H, W = x.size()
        return F.interpolate(y, size=(H, W), mode='nearest') + x

    def forward(self, x, params=None):
        c1 = F.relu(self.bn1(self.conv1(x, params=self.get_subdict(params, 'conv1')), 
                            params=self.get_subdict(params, 'bn1')))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        
        c2 = self.layer1(c1, params=self.get_subdict(params, 'layer1'))
        c3 = self.layer2(c2, params=self.get_subdict(params, 'layer2'))
        c4 = self.layer3(c3, params=self.get_subdict(params, 'layer3'))
        c5 = self.layer4(c4, params=self.get_subdict(params, 'layer4'))
        
        p6 = self.conv6(c5, params=self.get_subdict(params, 'conv6'))
        p7 = self.conv7(F.relu(p6), params=self.get_subdict(params, 'conv7'))
        
        p5 = self.latlayer1(c5, params=self.get_subdict(params, 'latlayer1'))
        p4 = self._upsample_add(p5, self.latlayer2(c4, params=self.get_subdict(params, 'latlayer2')))
        p4 = self.toplayer1(p4, params=self.get_subdict(params, 'toplayer1'))
        p3 = self._upsample_add(p4, self.latlayer3(c3, params=self.get_subdict(params, 'latlayer3')))
        p3 = self.toplayer2(p3, params=self.get_subdict(params, 'toplayer2'))

        return p3, p7
def MetaFPN50():
    return MetaFPN(MetaBottleneck, [3, 4, 6, 3])

def MetaFPN101():
    return MetaFPN(MetaBottleneck, [2, 4, 23, 3])

class MetaTwoStreamFPN(MetaModule):
    def __init__(self, FPN=MetaFPN50):
        super(MetaTwoStreamFPN, self).__init__()
        self.fpn_rgb = FPN()
        self.fpn_tex = FPN()

    def forward(self, rfb_input, tex_input, params=None):
        rgb_params = self.get_subdict(params, 'fpn_rgb')
        tex_params = self.get_subdict(params, 'fpn_tex')
        
        p3_rgb, p7_rgb = self.fpn_rgb(rfb_input, params=rgb_params)
        p3_tex, p7_tex = self.fpn_tex(tex_input, params=tex_params)
        
        p3 = (p3_rgb + p3_tex) / 2
        p7 = (p7_rgb + p7_tex) / 2
        
        return p3, p7

class MetaHierachicalFusionNetwork(MetaModule):
    def __init__(self, mean_std_normalize=True, dropout_rate=0.0):
        super(MetaHierachicalFusionNetwork, self).__init__()
        self.fpn = MetaTwoStreamFPN()
        self.pred_head = self._make_head()
        self.pred_head_cls = MetaSequential(
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.cls_heads = MetaSequential(
            torch.nn.Linear(256, 2)
        )
        self.mean_std_normalize = mean_std_normalize
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        if mean_std_normalize:
            self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _make_head(self):
        return MetaSequential(
            MetaConv2d(256, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),
            MetaConv2d(128, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x_rgb, x_tex, params=None):
        if self.mean_std_normalize:
            normalized = []
            for i in range(x_tex.shape[0]):
                normalized.append(self.normalize(x_tex[i]))
            x_tex = torch.stack(normalized, 0)
        
        p3, p7 = self.fpn(x_rgb, x_tex, params=self.get_subdict(params, 'fpn'))
        
        map_pred = self.pred_head(p3, params=self.get_subdict(params, 'pred_head'))
        
        cls_out = self.pred_head_cls(p7, params=self.get_subdict(params, 'pred_head_cls'))
        cls_out = cls_out.view(cls_out.size(0), -1)
        cls_out = self.dropout(cls_out)
        cls_out = self.cls_heads(cls_out, params=self.get_subdict(params, 'cls_heads'))
        
        return [map_pred], [cls_out]

