import os
import time

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.dataset import RockDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
val_image_dir = os.path.join(base_dir, "valid", "images")
val_mask_dir = os.path.join(base_dir, "valid", "masks")
save_dir = "inference_results"
os.makedirs(save_dir, exist_ok=True)

val_dataset = RockDataset(
    val_image_dir,
    val_mask_dir,
    val_transform,
)

encoder_name = "efficientnet-b0"
model = smp.Unet(
    encoder_name=encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation="sigmoid",
)
model.load_state_dict(
    torch.load(f"weights/Unet_{encoder_name}.pth", map_location=device)
)
model.to(device)
model.eval()

# Perform inference on 10 images
num_images = 10
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

with torch.no_grad():
    for i in tqdm(range(min(num_images, len(val_dataset))), desc="Inference"):
        image, mask = val_dataset[i]
        image_tensor = image.unsqueeze(0).to(device)

        start_time = time.time()
        output = model(image_tensor)
        pred_mask = (output.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(
            np.uint8
        ) * 255
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Inference time for image {i + 1}: {end_time - start_time:.4f} seconds")

        original_image = image.permute(1, 2, 0).cpu().numpy()
        original_image = (original_image * std + mean) * 255  # Denormalize
        original_image = np.clip(original_image, 0, 255).astype(np.uint8)
        original_mask = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pred_mask_resized = cv2.resize(
            pred_mask, (original_image.shape[1], original_image.shape[0])
        )

        combined_image = np.hstack(
            (
                original_image,
                cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(pred_mask_resized, cv2.COLOR_GRAY2BGR),
            )
        )

        label_height = 30
        labeled_image = np.zeros(
            (combined_image.shape[0] + label_height, combined_image.shape[1], 3),
            dtype=np.uint8,
        )
        labeled_image[label_height:, :, :] = combined_image
        cv2.putText(
            labeled_image,
            "Original Image",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            labeled_image,
            "Original Mask",
            (original_image.shape[1] + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            labeled_image,
            "Predicted Mask",
            (2 * original_image.shape[1] + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        image_path = os.path.join(save_dir, f"result_{i + 1}.png")
        cv2.imwrite(image_path, labeled_image)

print(f"Inference completed. Results saved in '{save_dir}'.")
