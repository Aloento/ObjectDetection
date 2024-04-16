from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from DetectionHead import DetectionHead
from FeatureLayer import FeatureLayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_anchor = 1

        self.class_head = FeatureLayer()
        self.bbox_head = DetectionHead()

        self.ce = CrossEntropyLoss()

    def forward(self, x: Tensor, labels: Tensor) -> (Tensor, dict[str, Tensor], dict[str, Tensor]):
        class_head, feats = self.class_head(x)
        bbox_head = self.bbox_head(feats)

        target_class = labels[:, -1].long()
        loss = self.ce(class_head, target_class)
        return class_head, loss


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    inputs, labs = next(iter(val_loader))

    _, loss_cls = model(inputs.to(device), labs.to(device))
    loss_cls.backward()

    print("Loss Class:", loss_cls)
