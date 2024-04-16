import torchvision.models
from torch import nn, Tensor


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_anchor = 1

        self.res = torchvision.models.resnet101(pretrained=True)
        self.res = nn.Sequential(*list(self.res.children())[:-2])

        self.regressor = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 4)
        )

        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Identity(),
            nn.Linear(2048, 17)
        )

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, x: Tensor, labels: Tensor) -> (Tensor, dict[str, Tensor], dict[str, Tensor]):
        feat = self.res(x)

        bbox = self.regressor(feat)
        cls = self.class_head(feat)

        target_class = labels[:, -1].long()
        cls_loss = self.ce(cls, target_class)

        target_bbox = labels[:, :-1]
        bbox_loss = self.mse(bbox, target_bbox)

        return (cls, bbox), (cls_loss, bbox_loss)


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, val_loader = prepare()
    model = Model().to(device)
    model.train()

    inputs, labs = next(iter(val_loader))

    _, loss_cls, loss_box = model(inputs.to(device), labs.to(device))

    print("Loss Class:", loss_cls)
    print("Loss Box:", loss_box)

    loss_t = loss_cls + loss_box
    print("Total Loss:", loss_t)
    loss_t.backward()
