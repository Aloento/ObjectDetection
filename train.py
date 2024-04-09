import torch
from torch import optim
from tqdm import tqdm

from Model import Model
from prepare import prepare


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, leave=True, position=1, desc="Training")

    # torch.Size([batch_size, 3 RGB channels, 640 w, 640 h])
    # torch.Size([num_of_images, (x Center, y Center, Width, Height, Class) 5 dims])
    for images, targets in loop:  # type: torch.Tensor, torch.Tensor
        images = images.to(device)
        targets = targets.to(device)

        break

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    train_loader, val_loader, test_loader = prepare()

    model = Model().to(device)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    epochs = 100
    # writer = SummaryWriter()

    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = 0

        print(f"\nEpoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
