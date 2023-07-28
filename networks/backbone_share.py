import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
#from nets.transformer_noffn import Transformer
#from nets.transformer_noffn import Transformer
from torchvision import ops
import torch

#from nets.cbam import SpatialGate
from copy import deepcopy

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

class FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN, self).__init__()
        self.in_planes = 64

        print('basline')

        resnet_hand = resnet50(pretrained=pretrained)

        #resnet_obj = resnet50(pretrained=pretrained)
        resnet_obj = deepcopy(resnet_hand)

        self.toplayer_h = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.toplayer_o = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.layer0_h = nn.Sequential(resnet_hand.conv1, resnet_hand.bn1, resnet_hand.leakyrelu, resnet_hand.maxpool)
        self.layer1_h = nn.Sequential(resnet_hand.layer1)
        self.layer2_h = nn.Sequential(resnet_hand.layer2)
        self.layer3_h = nn.Sequential(resnet_hand.layer3)
        self.layer4_h = nn.Sequential(resnet_hand.layer4)

        #self.layer0_o = nn.Sequential(resnet_obj.conv1, resnet_obj.bn1, resnet_obj.leakyrelu, resnet_obj.maxpool)
        #self.layer1_o = nn.Sequential(resnet_obj.layer1)
        self.layer2_o = nn.Sequential(resnet_obj.layer2)
        self.layer3_o = nn.Sequential(resnet_obj.layer3)
        #self.layer4_o = nn.Sequential(resnet_obj.layer4)


        # Smooth layers
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #self.smooth2_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        #self.smooth2_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_h = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_h = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_h = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)


        self.latlayer1_o = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_o = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_o = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1_h = self.layer0_h(x)
        #c1_o = self.layer0_h(x)

        c2_h = self.layer1_h(c1_h)
        #c2_o = c2_h
        #c2_o = self.layer1_h(c1_o)
    
        c3_h = self.layer2_h(c2_h)
        c3_o = self.layer2_o(c2_h)
  
        c4_h = self.layer3_h(c3_h)
        c4_o = self.layer3_o(c3_o)
 
        c5_h = self.layer4_h(c4_h)
        c5_o = self.layer4_h(c4_o)
    
        # Top-down
        p5_h = self.toplayer_h(c5_h)
        p4_h = self._upsample_add(p5_h, self.latlayer1_h(c4_h))
        p3_h = self._upsample_add(p4_h, self.latlayer2_h(c3_h))
        p2_h = self._upsample_add(p3_h, self.latlayer3_h(c2_h))


        p5_o = self.toplayer_o(c5_o)
        p4_o = self._upsample_add(p5_o, self.latlayer1_o(c4_o))
        p3_o = self._upsample_add(p4_o, self.latlayer2_o(c3_o))
        p2_o = self._upsample_add(p3_o, self.latlayer3_o(c2_h))
        # Smooth
        #p4 = self.smooth1(p4)
        #p3_h = self.smooth2(p3_h)

        p2_h = self.smooth3_h(p2_h)
        p2_o = self.smooth3_o(p2_o)
        #print(p2.shape)
        

        
        return p2_h , p2_o

class FPN_18(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN_18, self).__init__()
        self.in_planes = 64

        resnet_hand = resnet18(pretrained=pretrained)

        #resnet_obj = resnet18(pretrained=pretrained)
        resnet_obj = deepcopy(resnet_hand)

        self.toplayer_h = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.toplayer_o = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.layer0_h = nn.Sequential(resnet_hand.conv1, resnet_hand.bn1, resnet_hand.leakyrelu, resnet_hand.maxpool)
        self.layer1_h = nn.Sequential(resnet_hand.layer1)
        self.layer2_h = nn.Sequential(resnet_hand.layer2)
        self.layer3_h = nn.Sequential(resnet_hand.layer3)
        self.layer4_h = nn.Sequential(resnet_hand.layer4)

       # self.layer0_o = nn.Sequential(resnet_obj.conv1, resnet_obj.bn1, resnet_obj.leakyrelu, resnet_obj.maxpool)
        #self.layer1_o = nn.Sequential(resnet_obj.layer1)
        self.layer2_o = nn.Sequential(resnet_obj.layer2)
        self.layer3_o = nn.Sequential(resnet_obj.layer3)
       # self.layer4_o = nn.Sequential(resnet_obj.layer4)


        # Smooth layers
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_h = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_h = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_h = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)


        self.latlayer1_o = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_o = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_o = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)


        self.pool_h = nn.AvgPool2d(2, stride=2)
        self.pool_o = nn.AvgPool2d(2, stride=2)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1_h = self.layer0_h(x)
        #c1_o = self.layer0_o(x)

        c2_h = self.layer1_h(c1_h)
       # c2_o = self.layer1_o(c1_o)
    
        c3_h = self.layer2_h(c2_h)
        c3_o = self.layer2_o(c2_h)
  
        c4_h = self.layer3_h(c3_h)
        c4_o = self.layer3_o(c3_o)
 
        c5_h = self.layer4_h(c4_h)
        c5_o = self.layer4_o(c4_o)
    
        # Top-down
        p5_h = self.toplayer_h(c5_h)
        p4_h = self._upsample_add(p5_h, self.latlayer1_h(c4_h))
        p3_h = self._upsample_add(p4_h, self.latlayer2_h(c3_h))
        p2_h = self._upsample_add(p3_h, self.latlayer3_h(c2_h))


        p5_o = self.toplayer_o(c5_o)
        p4_o = self._upsample_add(p5_o, self.latlayer1_o(c4_o))
        p3_o = self._upsample_add(p4_o, self.latlayer2_o(c3_o))
        p2_o = self._upsample_add(p3_o, self.latlayer3_o(c2_h))
        # Smooth


        p2_h = self.smooth3_h(p2_h)
        p2_o = self.smooth3_o(p2_o)
       
        
        return p2_h , p2_o


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model Encoder"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth"))
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out