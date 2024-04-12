from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from FeatureLayer import FeatureLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.res = FeatureLayer()
        self.ce = CrossEntropyLoss()

    def forward(self, x: Tensor, labels: Tensor) -> (Tensor, dict[str, Tensor], dict[str, Tensor]):
        outputs = self.res(x)
        loss = self.ce(outputs, labels)
        return outputs, loss


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    inputs, labels = next(iter(val_loader))

    loss_cls = model(inputs.to(device), labels.to(device))
    loss_cls.backward()

    print("Loss Class:", loss_cls)
