import time

import torch
from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import Model
from torch.utils.data import DataLoader

from ObjectDetectionDataset import ObjectDetectionDataset
from persist import load_checkpoint
from prepare import load_dicts

if __name__ == '__main__':
    statues_dict, bgs_dict = load_dicts()

    test_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="test")
    test_loader = DataLoader(test_dataset)

    model = Model()
    epoch = load_checkpoint(model)
    model.eval()

    loop = tqdm(test_loader, desc="Testing")
    writer = SummaryWriter(comment=f"Test {epoch}")

    predictions = []
    targets = []

    with torch.no_grad():
        for i, (image, target) in enumerate(loop):  # type: int, (torch.Tensor, torch.Tensor)
            start_t = time.time()
            (cls, bbox), (loss_cls, loss_bbox) = model(image, target)
            end_t = time.time()

            writer.add_scalar("Loss/Class", loss_cls.item(), i)
            writer.add_scalar("Loss/Box", loss_bbox.item(), i)
            writer.add_scalar("Time", end_t - start_t, i)

            image = image[0]
            pred_box = bbox[0]
            pred_box[2] += pred_box[0]
            pred_box[3] += pred_box[1]

            pred_label = cls[0].argmax()
            pred_score = torch.softmax(cls[0], dim=0).max()
            targ_label = target[0, -1]

            writer.add_image_with_boxes(
                tag=f'Test {i}',
                img_tensor=image,
                box_tensor=pred_box.unsqueeze(0),
                labels=[f"{pred_label.item()} / {pred_score.item():.2f} - {targ_label.item()}"],
                global_step=i
            )

            target_bbox = target[0][:-1]
            target_bbox[2] += target_bbox[0]
            target_bbox[3] += target_bbox[1]

            predictions.append((int(pred_label), pred_score, pred_box))
            targets.append((int(targ_label), target_bbox))

    TP: dict[int, int] = {}
    FP: dict[int, int] = {}
    FN: dict[int, int] = {}

    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    f1: dict[int, float] = {}

    for i in range(0, 17):
        TP[i] = 0
        FP[i] = 0
        FN[i] = 0
        precision[i] = 0
        recall[i] = 0
        f1[i] = 0

    for pred, targ in zip(predictions, targets):
        pred_label, pred_score, pred_box = pred
        targ_label, targ_box = targ

        if pred_label == targ_label:
            iou = box_iou(pred_box.unsqueeze(0), targ_box.unsqueeze(0)).item()
            if iou > 0.5:
                TP[pred_label] += 1
            else:
                FP[pred_label] += 1
        else:
            FN[targ_label] += 1

    for i in range(0, 17):
        precision[i] = TP[i] / (TP[i] + FP[i])
        recall[i] = TP[i] / (TP[i] + FN[i])
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        writer.add_scalar(f"Precision/{i}", precision[i], i)
        writer.add_scalar(f"Recall/{i}", recall[i], i)
        writer.add_scalar(f"F1/{i}", f1[i], i)



