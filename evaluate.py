import torch
from albumentations.core.bbox_utils import convert_bbox_from_albumentations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DetectionMetric import DetectionMetric
from Model import Model


def evaluate_epoch(
        model: Model,
        dataloader: DataLoader,
        device: torch.device,
        metric: DetectionMetric,
        writer: SummaryWriter,
        epoch: int
):
    model.eval()
    loop = tqdm(dataloader, leave=True, position=3, desc="Evaluation")

    last_image = None
    last_target = None
    last_prediction = None

    with torch.no_grad():
        for i, (images, bboxes) in enumerate(loop):
            images = images.to(device)
            bboxes = bboxes.to(device)

            _, pred, targ = model(images, bboxes)
            map_score, precision_score, recall_score, f1_score = metric(pred, targ)

            loop.set_postfix(
                map=map_score['map'].item(),
                precision=precision_score.item(),
                recall=recall_score.item(),
                f1=f1_score.item()
            )

            writer.add_scalar('Metrics/mAP', map_score['map'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/50', map_score['map_50'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/75', map_score['map_75'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/small', map_score['map_small'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/medium', map_score['map_medium'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/large', map_score['map_large'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/1', map_score['mar_1'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/10', map_score['mar_10'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/100', map_score['mar_100'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/small', map_score['mar_small'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/medium', map_score['mar_medium'], epoch * len(dataloader) + i)
            # writer.add_scalar('Metrics/mAP/large', map_score['mar_large'], epoch * len(dataloader) + i)
            #
            # for m, cls_idx in enumerate(map_score['classes']):
            #     writer.add_scalar(
            #         f'Metrics/mAP/per_class/{cls_idx}',
            #         map_score['map_per_class'][m], epoch * len(dataloader) + i)
            #     writer.add_scalar(
            #         f'Metrics/mAP/100_per_class/{cls_idx}',
            #         map_score['mar_100_per_class'][m], epoch * len(dataloader) + i)

            writer.add_scalar("Metrics/Precision", precision_score, epoch * len(dataloader) + i)
            writer.add_scalar("Metrics/Recall", recall_score, epoch * len(dataloader) + i)
            writer.add_scalar("Metrics/F1", f1_score, epoch * len(dataloader) + i)

            last_image = images[0]
            last_target = targ
            last_prediction = pred

    pred_box = last_prediction["boxes"][0]
    pred_box_xyxy = convert_bbox_from_albumentations(
        bbox=pred_box,
        target_format='pascal_voc',
        rows=640,
        cols=640
    )

    pred_label = last_prediction["labels"][0].item()
    pred_score = last_prediction["scores"][0].item()

    target_box = last_target["boxes"][0]
    target_box_xyxy = convert_bbox_from_albumentations(
        bbox=target_box,
        target_format='pascal_voc',
        rows=640,
        cols=640
    )

    box_xyxy = torch.tensor([pred_box_xyxy, target_box_xyxy])

    target_label = last_target["labels"][0].item()

    writer.add_image_with_boxes(
        tag='Prediction',
        img_tensor=last_image,
        box_tensor=box_xyxy,
        labels=[f'Pred: {pred_label} / {pred_score}', f'Target: {target_label}'],
        global_step=epoch
    )

    return