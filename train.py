import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import Model
from persist import save_checkpoint, load_checkpoint
from prepare import prepare


def train_epoch(
        model: Model,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        scaler: GradScaler,
        writer: SummaryWriter,
        epoch: int
) -> float:
    model.train()
    total_loss = 0  # type: float
    loop = tqdm(dataloader, desc="Training")

    for i, (images, targets) in enumerate(loop):  # type: int, (torch.Tensor, torch.Tensor)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            _, (loss_cls, loss_bbox) = model(images, targets)

        loss_total = loss_cls + loss_bbox
        total_loss += loss_total.item()

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(
            loss_cls=loss_cls.item(),
            loss_box=loss_bbox.item()
        )

        current = epoch * len(dataloader) + i
        writer.add_scalar("Loss/Class", loss_cls.item(), current)
        writer.add_scalar("Loss/Box", loss_bbox.item(), current)

    return total_loss / len(dataloader)


def validate_epoch(
        model: Model,
        dataloader: DataLoader,
        device: torch.device,
        writer: SummaryWriter,
        epoch: int
) -> float:
    model.eval()
    total_loss = 0  # type: float
    loop = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for i, (images, targets) in enumerate(loop):  # type: int, (torch.Tensor, torch.Tensor)
            images = images.to(device)
            targets = targets.to(device)

            (cls, bbox), (loss_cls, loss_bbox) = model(images, targets)

            loss_total = loss_cls + loss_bbox
            total_loss += loss_total.item()

            loop.set_postfix(
                loss_cls=loss_cls.item(),
                loss_box=loss_bbox.item()
            )

            current = epoch * len(dataloader) + i
            writer.add_scalar("Loss/Validation/Class", loss_cls.item(), current)
            writer.add_scalar("Loss/Validation/Box", loss_bbox.item(), current)

            image = images[0]
            pred_box = bbox[0]
            pred_box[2] += pred_box[0]
            pred_box[3] += pred_box[1]

            pred_label = cls[0].argmax().item()
            pred_score = cls[0].max().item()
            targ_label = targets[0, -1].item()

            writer.add_image_with_boxes(
                tag=f'Prediction {i}',
                img_tensor=image,
                box_tensor=pred_box.unsqueeze(0),
                labels=[f"{pred_label} / {pred_score:.2f} - {targ_label}"],
            )

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    train_loader, val_loader = prepare()

    model = Model().to(device)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = GradScaler()

    start_epoch = load_checkpoint(model, optimizer, scheduler)
    epochs = 1000
    writer = SummaryWriter()

    prog = tqdm(range(start_epoch, epochs), desc="Epochs")

    for epoch in prog:
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, writer, epoch)

        val_loss = validate_epoch(model, val_loader, device, writer, epoch)
        scheduler.step(val_loss)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        prog.set_postfix(
            train=train_loss,
            val=val_loss
        )

        print(f"\nEpoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch)

    writer.close()


if __name__ == "__main__":
    main()
