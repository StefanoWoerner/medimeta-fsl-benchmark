import os
from itertools import repeat
import pandas as pd

import torch
from torchcross.models.lightning import SimpleClassifier

from models.tranformers_backbones import get_backbone_from_name

from lightning import pytorch as pl

backbones = [
    "google/vit-base-patch16-224",
    "google/vit-base-patch16-224-in21k",
    "openai/clip-vit-base-patch16",
    "ikim-uk-essen/BiomedCLIP_ViT_patch16_224",
    "facebook/dinov2-base",
    "microsoft/resnet-18",
    "microsoft/resnet-50",
    "microsoft/resnet-101",
    "openai/clip-vit-large-patch14",
    "google/vit-large-patch16-224",
    "google/vit-large-patch16-224-in21k",
    "facebook/dinov2-large",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "google/vit-huge-patch14-224-in21k",
    "facebook/dinov2-giant",
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    # "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
]

num_train_samples = [
    1_281_167,
    14_197_122,
    400_000_000,
    15_282_336,
    142_000_000,
    1_281_167,
    1_281_167,
    1_281_167,
    400_000_000,
    1_281_167,
    14_197_122,
    142_000_000,
    2_000_000_000,
    14_197_122,
    142_000_000,
    2_000_000_000,
    2_000_000_000,
]

medical_data = [
    "no",
    "no",
    "maybe",
    "yes",
    "maybe",
    "no",
    "no",
    "no",
    "maybe",
    "no",
    "no",
    "maybe",
    "maybe",
    "no",
    "maybe",
    "maybe",
    "maybe",
]

backbone_label_map = {
    "google/vit-base-patch16-224": "ViT-B/16",
    "google/vit-base-patch16-224-in21k": "ViT-B/16-IN21k",
    "openai/clip-vit-base-patch16": "CLIP-ViT-B/16",
    "ikim-uk-essen/BiomedCLIP_ViT_patch16_224": "BiomedCLIP-ViT-B/16",
    "facebook/dinov2-base": "DINOv2-ViT-B/14",
    "microsoft/resnet-18": "ResNet-18",
    "microsoft/resnet-50": "ResNet-50",
    "microsoft/resnet-101": "ResNet-101",
    "openai/clip-vit-large-patch14": "CLIP-ViT-L/14",
    "google/vit-large-patch16-224": "ViT-L/16",
    "google/vit-large-patch16-224-in21k": "ViT-L/16-IN21k",
    "facebook/dinov2-large": "DINOv2-ViT-L/14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "CLIP-ViT-H/14-laion2B",
    "google/vit-huge-patch14-224-in21k": "ViT-H/14-IN21k",
    "facebook/dinov2-giant": "DINOv2-ViT-g/14",
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K": "CLIP-ViT-g/14-laion2B",
    # "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": "CLIP-ViT-G/14",
}


records = []

for backbone_name, n_tr, med in zip(backbones, num_train_samples, medical_data):

    backbone, num_backbone_features = get_backbone_from_name(backbone_name)

    print(
        f"Number of parameters in backbone {backbone_name} with {num_backbone_features} features:"
    )
    print(sum(p.numel() for p in backbone.parameters()))

    records.append(
        {
            "backbone_name": backbone_name,
            "backbone": backbone_label_map[backbone_name],
            "num_backbone_features": num_backbone_features,
            "num_parameters": sum(p.numel() for p in backbone.parameters()),
            "num_train_samples": n_tr,
            "medical_data": med,
        }
    )

df = pd.DataFrame(records)
df.to_csv("backbone_parameters.csv")
print(df)
