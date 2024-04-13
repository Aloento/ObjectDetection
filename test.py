from ProcessBBox import process_boxes
from prepare import prepare

from Model import Model

if __name__ == "__main__":
    _, val_loader = prepare()

    images, bboxes, labels = next(iter(val_loader))
    print(images.shape, len(bboxes), labels.shape)

    model = Model()
    model.train()

    deep, medium, shallow = model(images)
    print(deep.shape, medium.shape, shallow.shape)

    model.eval()
    pred = model.predict(deep, medium, shallow)
    print(pred.shape)

    bboxes = process_boxes(pred)
    print(bboxes)
