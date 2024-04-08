from torch import nn

from ComputeLoss import ComputeLoss
from DetectionLayer import DetectionLayer
from FeatureExtractionLayer import FeatureExtractionLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fe = FeatureExtractionLayer()
        self.out = DetectionLayer()

    def forward(self, x):
        x = self.fe(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    from prepare import prepare

    _, _, test_loader = prepare()
    model = Model()

    image, target = next(iter(test_loader))
    output = model(image)

    print("Output shape:", output.shape)
    print("Target shape:", target.shape)

    loss = ComputeLoss()(output, target)
