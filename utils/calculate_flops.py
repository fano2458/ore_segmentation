import csv

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import encoders, get_encoder_names
from torchinfo import summary
from tqdm import tqdm

model_names = get_encoder_names()

csv_file = "model_metrics.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model Name", "FLOPs", "Parameters"])

    for model_name in tqdm(model_names):
        available_weights = list(encoders[model_name]["pretrained_settings"].keys())
        pretrained_weights = (
            "imagenet" if "imagenet" in available_weights else available_weights[0]
        )

        try:
            model = smp.Unet(
                encoder_name=model_name,
                encoder_weights=pretrained_weights,
                in_channels=3,
                classes=1,
                activation="sigmoid",
            )
        except:
            print(
                f"Failed to create model {model_name} with weights {pretrained_weights}"
            )
            continue

        input_tensor = torch.randn(1, 3, 1024, 1024)
        model_info = summary(model, input_data=input_tensor, verbose=0)
        flops = model_info.total_mult_adds
        params = model_info.total_params

        writer.writerow([model_name, flops, params])
