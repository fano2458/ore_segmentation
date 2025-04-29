import segmentation_models_pytorch as smp
import torch
from torchinfo import summary

model_fastest = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid",
).eval()

model_tradeoff = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid",
).eval()

model_strongest = smp.DeepLabV3Plus(
    encoder_name="mit_b5",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid",
).eval()


models = [model_fastest, model_tradeoff, model_strongest]

input_tensor = torch.randn(1, 3, 1920, 1088)
for model in models:
    model_info = summary(model, input_data=input_tensor, verbose=1)
    flops = model_info.total_mult_adds
    params = model_info.total_params
    print(f"Model: {model.__class__.__name__}, FLOPs: {flops}, Parameters: {params}")
