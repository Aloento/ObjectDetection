from torch import nn, Tensor

from ComputeLoss import ComputeLoss
from DetectionLayer import DetectionLayer
from FeatureExtractionLayer import FeatureExtractionLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fe = FeatureExtractionLayer()
        self.out = DetectionLayer()
        self.loss = ComputeLoss()

    def forward(self, images: Tensor, bboxes: Tensor) -> Tensor:
        images = self.fe(images)
        images = self.out(images)

        loss = self.loss(images, bboxes)
        return loss


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    image, target = next(iter(val_loader))
    comp_loss = model(image.to(device), target.to(device))
    comp_loss.backward()

    print("Loss:", comp_loss)
