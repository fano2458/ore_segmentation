import gc
import os

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from tqdm import tqdm

import wandb
from src.dataset import RockDataset

torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 100
print(f"Using device: {device}")

encoder_name = "efficientnet-b0"
# encoder_name = "mobilenet_v2"
# encoder_name = "mobileone_s0"
# encoder_name = "mit_b0"
# encoder_name = "resnet18"

train_transform = A.Compose(
    [
        A.PadIfNeeded(
            min_height=1088,
            min_width=1920,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomCrop(height=512, width=512, p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-15, 15), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.PadIfNeeded(
            min_height=1088,
            min_width=1920,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
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
    batch_size=1,
    shuffle=True,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

model = smp.UnetPlusPlus(
    encoder_name=encoder_name,
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
)

model.to(device)

accumulation_steps = 16
best_val_f1 = 0.0
save_path = f"weights/Unet_{encoder_name}.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

patience = 5  # Early stopping patience
no_improvement_epochs = 0  # Counter for epochs without improvement

wandb.init(project="ore_segmentation", name=f"unet++_{encoder_name}")

# Log model configuration
wandb.config.update(
    {
        "encoder_name": encoder_name,
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": 1e-4,
        "patience": patience,
    }
)


def denormalize_image(image):
    """Denormalize image for visualization."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean) * 255.0
    return image.clip(0, 255).astype("uint8")


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for i, (images, masks) in tqdm(
        enumerate(train_loader), desc="Training", total=len(train_loader)
    ):
        images = images.to(device)
        masks = masks.to(device).float()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
    val_accuracy = 0.0
    val_precision = 0.0
    val_recall = 0.0
    predictions_to_log = []  # Store predictions for logging
    with torch.no_grad():
        for idx, (images, masks) in enumerate(
            tqdm(val_loader, desc="Validation", total=len(val_loader))
        ):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            val_loss += loss.item() * images.size(0)

            preds = outputs.squeeze(1) > 0.5
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            f1 = f1_score(
                masks_np.flatten(),
                preds_np.flatten(),
                average="binary",
                zero_division=0,
            )
            val_f1 += f1 * images.size(0)

            # Calculate additional metrics
            tp = ((preds_np == 1) & (masks_np == 1)).sum()
            tn = ((preds_np == 0) & (masks_np == 0)).sum()
            fp = ((preds_np == 1) & (masks_np == 0)).sum()
            fn = ((preds_np == 0) & (masks_np == 1)).sum()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            val_accuracy += accuracy * images.size(0)
            val_precision += precision * images.size(0)
            val_recall += recall * images.size(0)

            # Save predictions for logging every epoch
            if len(predictions_to_log) < 3:
                denormalized_image = denormalize_image(images[0])
                original_mask = masks[0].cpu().numpy()
                predicted_mask = preds[0].cpu().numpy()
                predictions_to_log.append(
                    wandb.Image(
                        denormalized_image,
                        masks={
                            "ground_truth": {"mask_data": original_mask},
                            "prediction": {"mask_data": predicted_mask},
                        },
                        caption="Original Image, Ground Truth Mask, Predicted Mask",
                    )
                )

            del images, masks, outputs, loss, preds
            gc.collect()
            torch.cuda.empty_cache()

    val_loss /= len(val_loader.dataset)
    val_f1 /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)
    val_precision /= len(val_loader.dataset)
    val_recall /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
    )

    # Log metrics to wandb
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
        }
    )

    # Log predictions every epoch
    wandb.log({"predictions": predictions_to_log})

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model with Val F1: {best_val_f1:.4f}")
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        print(f"No improvement for {no_improvement_epochs} epoch(s).")

    if no_improvement_epochs >= patience:
        print("Early stopping triggered.")
        break

    gc.collect()
    torch.cuda.empty_cache()

wandb.finish()
