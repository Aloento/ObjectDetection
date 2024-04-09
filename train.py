import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import Model
from prepare import prepare


def train_epoch(
        model: Model,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device
) -> float:
    model.train()
    total_loss = 0  # type: float
    loop = tqdm(dataloader, leave=True, position=1, desc="Training")

    # torch.Size([batch_size, 3 RGB channels, 640 w, 640 h])
    # torch.Size([num_of_images, (x Center, y Center, Width, Height, Class) 5 dims])
    for images, bboxes in loop:  # type: torch.Tensor, torch.Tensor
        images = images.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        loss = model(images, bboxes)  # type: torch.Tensor
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate_epoch(
        model: Model,
        dataloader: DataLoader,
        device: torch.device
) -> float:
    model.eval()
    total_loss = 0  # type: float
    loop = tqdm(dataloader, leave=True, position=2, desc="Validation")

    with torch.no_grad():
        for images, bboxes in loop:  # type: torch.Tensor, torch.Tensor
            images = images.to(device)
            bboxes = bboxes.to(device)

            loss = model(images, bboxes)  # type: torch.Tensor
            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    train_loader, val_loader = prepare()

    model = Model().to(device)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    epochs = 100
    writer = SummaryWriter()

    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        val_loss = 0
        scheduler.step(val_loss)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        print(f"\nEpoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
