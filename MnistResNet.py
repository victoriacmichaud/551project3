import torch
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class MnistResNet(ResNet):
    """ ResNet adapted for single (from 3) channel images
    
    https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial
    """
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(
            super(MnistResNet, self).forward(x), dim=-1)

class MnistResNet50(ResNet):
    """ ResNet adapted for single (from 3) channel images
    
    https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial
    """
    def __init__(self):
        super(MnistResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(
            super(MnistResNet50, self).forward(x), dim=-1)

class MnistResNet101(ResNet):
    """ ResNet adapted for single (from 3) channel images
    
    https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial
    """
    def __init__(self):
        super(MnistResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(
            super(MnistResNet101, self).forward(x), dim=-1)

class MnistResNet152(ResNet):
    """ ResNet adapted for single (from 3) channel images
    
    https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial
    """
    def __init__(self):
        super(MnistResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(
            super(MnistResNet152, self).forward(x), dim=-1)
