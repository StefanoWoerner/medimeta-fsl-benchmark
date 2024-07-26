import os
import argparse

import find_hparams2

test_dir = "./experiments/find_hparams2"

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
]

ns = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
n_query = 10
metatest_length = 10
linear_probe = True


if __name__ == "__main__":
    for n_support in ns:
        for backbone in backbones:
            print(f"Running {backbone} hyperparameter search on {n_support}-shot tasks")

            args = argparse.Namespace(
                data_path="data/MedIMeta",
                presampled_data_path="data/MedIMeta_presampled",
                backbone=backbone,
                linear_probe=linear_probe,
                n_support=n_support,
                n_query=n_query,
                metatest_length=metatest_length,
                test_dir=test_dir,
            )
            find_hparams2.main(args)
