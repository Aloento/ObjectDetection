from torch import nn, Tensor

from ComputeLoss import ComputeLoss
from FCLayer import FCLayer
from FeatureLayer import FeatureLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fe = FeatureLayer()
        self.out = FCLayer(512)
        self.loss = ComputeLoss()

    def forward(self, x: Tensor, bboxes: list[Tensor]) -> (Tensor, dict[str, Tensor], dict[str, Tensor]):
        x = self.fe(x)
        x = self.out(x)

        x = self.loss(x, bboxes)
        return x


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    image, boxes = next(iter(val_loader))
    boxes = [bbox.to(device) for bbox in boxes]

    loss_cls = model(image.to(device), boxes)
    loss_cls.backward()

    print("Loss Class:", loss_cls)
