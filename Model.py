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

    def forward(self, images: Tensor, bboxes: Tensor) -> (Tensor, dict[str, Tensor], dict[str, Tensor]):
        images = self.fe(images)
        images = self.out(images)

        loss, pred, targ = self.loss(images, bboxes)
        return loss, pred, targ


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
    comp_loss, pred, targ = model(image.to(device), target.to(device))
    comp_loss.backward()

    print("Loss:", comp_loss)

    model.eval()
    metric = DetectionMetric().to(device)
    map_score, precision_score, recall_score, f1_score = metric(pred, targ)

    print("MAP:", map_score)
    print("Precision:", precision_score)
    print("Recall:", recall_score)
    print("F1:", f1_score)