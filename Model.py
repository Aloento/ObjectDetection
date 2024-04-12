from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss

from FeatureLayer import FeatureLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.res = FeatureLayer()
        self.bce = BCEWithLogitsLoss()

    def forward(self, x: Tensor, bboxes: list[Tensor], labels: list[Tensor]):
        x = self.res(x)
        loss_cls = self.bce(x, labels)
        return x, loss_cls


if __name__ == "__main__":
    import torch
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    image, bbox_s, label_s = next(iter(val_loader))
    image = image.to(device)
    label_s = label_s.to(device)

    outputs, loss = model(image.to(device), bbox_s, label_s)
    loss.backward()
    print("Loss Class:", loss)

    predictions = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predictions == label_s).float().mean()
    print("Accuracy:", 100 * correct)
