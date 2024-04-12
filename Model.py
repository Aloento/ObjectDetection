from torch import nn, Tensor

from ComputeLoss import ComputeLoss
from FeatureLayer import FeatureLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.res = FeatureLayer()
        self.loss = ComputeLoss()

    def forward(self, x: Tensor, bboxes: list[Tensor], labels: list[Tensor]):
        x = self.res(x)
        x = self.loss(x, bboxes, labels)
        return x


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    image, bbox_s, label_s = next(iter(val_loader))

    loss_cls = model(image.to(device), bbox_s, label_s)
    loss_cls.backward()

    print("Loss Class:", loss_cls)
