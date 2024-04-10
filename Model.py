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

    def forward(self, predictions: Tensor, bboxes: Tensor) -> (Tensor, dict[str, Tensor], dict[str, Tensor]):
        predictions = self.fe(predictions)
        predictions = self.out(predictions)

        x = self.loss(predictions, bboxes)
        return x


if __name__ == "__main__":
    from prepare import prepare
    from torch.cuda import is_available
    from DetectionMetric import DetectionMetric

    device = "cuda" if is_available() else "cpu"
    print("Using device", device)

    _, _, test_loader = prepare()
    model = Model().to(device)
    model.train()

    image, target = next(iter(test_loader))
    loss_cls, (pred, targ) = model(image.to(device), target.to(device))
    loss_cls.backward()

    # print("Loss:", comp_loss)
    # print("Loss Box:", loss_box)
    print("Loss Class:", loss_cls)

    model.eval()
    metric = DetectionMetric().to(device)
    map_score, precision_score, recall_score, f1_score = metric(pred, targ)

    print("MAP:", map_score)
    print("Precision:", precision_score)
    print("Recall:", recall_score)
    print("F1:", f1_score)
