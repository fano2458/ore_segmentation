import gc
import os

import albumentations as A
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score

from src.dataset import RockDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 30
print(f"Using device: {device}")

train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

base_dir = "data"
train_image_dir = os.path.join(base_dir, "train", "images")
train_mask_dir = os.path.join(base_dir, "train", "masks")
val_image_dir = os.path.join(base_dir, "valid", "images")
val_mask_dir = os.path.join(base_dir, "valid", "masks")

train_dataset = RockDataset(
    train_image_dir,
    train_mask_dir,
    train_transform,
)
val_dataset = RockDataset(
    val_image_dir,
    val_mask_dir,
    val_transform,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
)

model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid",
)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
    verbose=True,
)

model.to(device)
torch.compile(model, device=device)

accumulation_steps = 8
best_val_f1 = 0.0
save_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device).float()

        outputs = model(images)
        loss = criterion(outputs.squeeze(1), masks)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * images.size(0) * accumulation_steps

        del images, masks, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            val_loss += loss.item() * images.size(0)

            preds = outputs.squeeze(1) > 0.5
            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()
            f1 = f1_score(masks, preds, average="binary", zero_division=0)
            val_f1 += f1 * images.size(0)

            del images, masks, outputs, loss, preds
            gc.collect()
            torch.cuda.empty_cache()

    val_loss /= len(val_loader.dataset)
    val_f1 /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model with Val F1: {best_val_f1:.4f}")

    gc.collect()
    torch.cuda.empty_cache()
