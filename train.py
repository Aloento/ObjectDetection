import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import Model
from persist import save_checkpoint, load_checkpoint
from prepare import prepare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
        model: Model,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        writer: SummaryWriter,
        epoch: int
) -> float:
    model.train()
    total_loss = 0  # type: float
    loop = tqdm(dataloader, leave=True, position=1, desc="Training")

    for i, (images, bboxes, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs, loss_cls = model(images, bboxes, labels)

        scaler.scale(loss_cls).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_cls.item()
        loop.set_postfix(loss=loss_cls.item())

        current = epoch * len(dataloader) + i

        if i % 10 == 0:
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predictions == labels).float().mean()

            writer.add_scalar("Accuracy", 100 * correct, current)
            writer.add_scalar("Loss/Class", loss_cls.item(), current)

    return total_loss / len(dataloader)


def validate_epoch(
        model: Model,
        dataloader: DataLoader,
        writer: SummaryWriter,
        epoch: int
) -> float:
    model.eval()
    total_loss = 0  # type: float
    loop = tqdm(dataloader, leave=True, position=2, desc="Validation")

    with torch.no_grad():
        for i, (images, bboxes, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            _, loss_cls = model(images, bboxes, labels)

            total_loss += loss_cls.item()
            loop.set_postfix(loss=loss_cls.item())

            if i % 10 == 0:
                writer.add_scalar("Loss/Validation Batch", loss_cls.item(), epoch * len(dataloader) + i)

    return total_loss / len(dataloader)


def main():
    print("Using device", device)

    train_loader, val_loader = prepare()

    model = Model().to(device)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = GradScaler()

    start_epoch = load_checkpoint(model, optimizer, scheduler)
    epochs = 200
    writer = SummaryWriter()

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", position=0):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, writer, epoch)

        val_loss = validate_epoch(model, val_loader, writer, epoch)
        scheduler.step(val_loss)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        print(f"\nEpoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch)

    writer.close()


if __name__ == "__main__":
    main()
