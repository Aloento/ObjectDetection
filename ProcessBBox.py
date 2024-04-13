import numpy as np
import torch
import torchvision
from torch import Tensor


def process_boxes(pred_boxes: Tensor):
    valid_scale = [0, np.inf]
    image_size = 416

    pred_boxes = pred_boxes.detach().cpu().numpy()
    batch_size, num_boxes, properties = pred_boxes.shape

    result = []

    for i in range(batch_size):
        current = pred_boxes[i, :, :]
        xy_wh = current[:, :4]
        confs = current[:, 4]
        probs = current[:, 5:]

        # (x, y, w, h) -> (x1, y1, x2, y2)
        coordinates = np.concatenate(
            [
                xy_wh[:, :2] - xy_wh[:, 2:] / 2,
                xy_wh[:, :2] + xy_wh[:, 2:] / 2
            ],
            axis=-1
        )

        # remove out of range boxes
        coordinates = np.concatenate(
            [
                np.maximum(coordinates[:, :2], [0, 0]),
                np.minimum(coordinates[:, 2:], [image_size - 1, image_size - 1])
            ],
            axis=-1
        )
        ranges_mask = np.logical_or(
            (coordinates[:, 0] > coordinates[:, 2]),
            (coordinates[:, 1] > coordinates[:, 3])
        )
        coordinates[ranges_mask] = 0

        # remove small / large boxes
        scales = np.sqrt(
            np.multiply.reduce(coordinates[:, 2:4] - coordinates[:, :2], axis=-1)
        )
        scales_mask = np.logical_and(
            (scales > valid_scale[0]),
            (scales < valid_scale[1])
        )

        # remove low confidence boxes
        classes = np.argmax(probs, axis=-1)
        scores = confs * probs[np.arange(len(coordinates)), classes]
        scores_mask = scores > 0.3
        mask = np.logical_and(scales_mask, scores_mask)

        coordinates, scores, classes = coordinates[mask], scores[mask], classes[mask]
        keep = torchvision.ops.nms(
            torch.tensor(coordinates),
            torch.tensor(scores),
            0.5
        ).numpy()

        coordinates, scores, classes = coordinates[keep], scores[keep], classes[keep]

        result.append(
            np.concatenate([coordinates, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        )

    return result
