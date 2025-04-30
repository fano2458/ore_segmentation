import argparse
import gc
import os

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

import wandb
from src.dataset import RockDataset

torch.set_float32_matmul_precision("high")


def denormalize_image(image):
    """Denormalize image for visualization."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean) * 255.0
    return image.clip(0, 255).astype("uint8")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_dir = "data"
    train_image_dir = os.path.join(base_dir, "train", "images")
    train_mask_dir = os.path.join(base_dir, "train", "masks")
    val_image_dir = os.path.join(base_dir, "valid", "images")
    val_mask_dir = os.path.join(base_dir, "valid", "masks")

    if args.model == "unet":
        encoder_name = args.encoder_name if args.encoder_name else "mobilenet_v2"
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
    elif args.model == "unetplusplus":
        encoder_name = args.encoder_name if args.encoder_name else "efficientnet-b4"
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
    elif args.model == "deeplabv3plus":
        encoder_name = args.encoder_name if args.encoder_name else "mit_b5"
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
    else:
        raise ValueError(
            "Invalid model type. Choose from 'unet', 'unetplusplus', or 'deeplabv3plus'."
        )

    model.to(device)

    train_transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=1088,
                min_width=1920,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                    A.OpticalDistortion(distort_limit=0.5, p=0.3),
                ],
                p=0.3,
            ),
            A.Affine(
                translate_percent=0.05,
                scale=(0.95, 1.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomCrop(height=512, width=512, p=1.0),
                    A.RandomResizedCrop(
                        size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0
                    ),
                ],
                p=0.5,
            ),
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs - args.warmup_epochs,
        eta_min=1e-7,
    )

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warmup_epochs,
        after_scheduler=cosine_scheduler,
    )

    best_val_f1 = 0.0
    no_improvement_epochs = 0

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    wandb.init(project="ore_segmentation_v1", name=f"{args.model}_{encoder_name}")

    wandb.config.update(
        {
            "model": args.model,
            "encoder_name": encoder_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
        }
    )

    # Maintain a list of top-3 models based on F1 score
    top_models = []

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        train_images_to_log = []  # Store training images and masks for logging
        for i, (images, masks) in tqdm(
            enumerate(train_loader), desc="Training", total=len(train_loader)
        ):
            images = images.to(device)
            masks = masks.to(device).float()

            if device == "cuda":
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(images)
            else:
                outputs = model(images)

            loss_bce = bce_loss(outputs.squeeze(1), masks)
            loss_dice = dice_loss(outputs.squeeze(1), masks)
            loss = (0.5 * (loss_bce + loss_dice)) / args.accumulation_steps
            loss.backward()

            # Handle gradient accumulation and remainder batches
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_bce_loss = loss_bce.item() * images.size(0) * args.accumulation_steps
            train_dice_loss = (
                loss_dice.item() * images.size(0) * args.accumulation_steps
            )
            train_loss += loss.item() * images.size(0) * args.accumulation_steps

            # Save images and masks for logging every epoch
            if len(train_images_to_log) < 3:
                denormalized_image = denormalize_image(images[0])
                original_mask = masks[0].cpu().numpy()
                train_images_to_log.append(
                    wandb.Image(
                        denormalized_image,
                        masks={
                            "ground_truth": {"mask_data": original_mask},
                        },
                        caption="Original Image and Ground Truth Mask",
                    )
                )

        train_bce_loss /= len(train_loader.dataset)
        train_dice_loss /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_bce_loss = 0.0
        val_dice_loss = 0.0
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

                val_loss_bce = bce_loss(outputs.squeeze(1), masks)
                val_loss_dice = dice_loss(outputs.squeeze(1), masks)
                val_loss += (
                    0.5 * (val_loss_bce.item() + val_loss_dice.item()) * images.size(0)
                )
                val_bce_loss += val_loss_bce.item() * images.size(0)
                val_dice_loss += val_loss_dice.item() * images.size(0)

                preds = outputs.squeeze(1) >= 0.5
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()

                f1 = f1_score(
                    masks_np.flatten(),
                    preds_np.flatten(),
                    average="binary",
                    zero_division=0,
                )
                val_f1 += f1 * images.size(0)

                # Calculate additional metrics with safeguards for division by zero
                tp = ((preds_np == 1) & (masks_np == 1)).sum()
                tn = ((preds_np == 0) & (masks_np == 0)).sum()
                fp = ((preds_np == 1) & (masks_np == 0)).sum()
                fn = ((preds_np == 0) & (masks_np == 1)).sum()

                accuracy = (
                    (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                )
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

        val_loss /= len(val_loader.dataset)
        val_bce_loss /= len(val_loader.dataset)
        val_dice_loss /= len(val_loader.dataset)
        val_f1 /= len(val_loader.dataset)
        val_accuracy /= len(val_loader.dataset)
        val_precision /= len(val_loader.dataset)
        val_recall /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val F1: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
        )

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_bce_loss": train_bce_loss,
                "train_dice_loss": train_dice_loss,
                "val_loss": val_loss,
                "val_bce_loss": val_bce_loss,
                "val_dice_loss": val_dice_loss,
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_images": train_images_to_log,
                "predictions": predictions_to_log,
            }
        )

        # Save top-3 models based on F1 score
        if len(top_models) < 3 or val_f1 > min(top_models, key=lambda x: x[0])[0]:
            model_path = os.path.join(
                weights_dir, f"{args.model}_{encoder_name}_valf1_{val_f1:.4f}.pth"
            )
            torch.save(model.state_dict(), model_path)
            top_models.append((val_f1, model_path))
            top_models = sorted(top_models, key=lambda x: x[0], reverse=True)[:3]
            print(f"Saved model with Val F1: {val_f1:.4f}")

            # Remove models that are no longer in the top-3
            for _, path in top_models[3:]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Removed model: {path}")

        # Early stopping logic
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs} epoch(s).")

        if no_improvement_epochs >= args.patience:
            print("Early stopping triggered.")
            break

        scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

    # Average weights of top-3 models
    if len(top_models) > 0:
        print("Averaging weights of top-3 models...")
        averaged_weights = None
        for _, model_path in top_models:
            state_dict = torch.load(model_path)
            if averaged_weights is None:
                averaged_weights = {k: v.clone() for k, v in state_dict.items()}
            else:
                for k in averaged_weights:
                    averaged_weights[k] += state_dict[k]
        for k in averaged_weights:
            averaged_weights[k] /= len(top_models)
        averaged_model_path = os.path.join(
            weights_dir, f"{args.model}_{encoder_name}_averaged.pth"
        )
        torch.save(averaged_weights, averaged_model_path)
        print(f"Saved averaged model weights to {averaged_model_path}.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument(
        "--model",
        type=str,
        help="Model type (unet, unetplusplus, deeplabv3plus)",
        required=True,
    )
    parser.add_argument("--encoder_name", type=str, help="Encoder name for the model")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )

    args = parser.parse_args()

    main(args)
