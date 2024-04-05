from torch import nn

from FEBlock1 import FEBlock1
from FEBlock2 import FEBlock2
from FEBlock3 import FEBlock3
from FEBlock4 import FEBlock4


class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()
        self.block1 = FEBlock1()
        self.block2 = FEBlock2()
        self.block3 = FEBlock3()
        self.block4 = FEBlock4()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x, residual = self.block3(x)
        x = self.block4(x, residual)
        return x
