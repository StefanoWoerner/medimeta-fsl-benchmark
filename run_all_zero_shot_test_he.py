import os
import argparse

import zero_shot_test_he

test_dir = "./experiments/zero-shot_test2"

datasets = [
    "aml",
    "bus",
    "cxr",
    "crc",
    "derm",
    "dr_regular",
    "dr_uwf",
    "fundus",
    "glaucoma",
    "mammo_mass",
    "mammo_calc",
    "oct",
    "organs_axial",
    "organs_coronal",
    "organs_sagittal",
    "pbc",
    "pneumonia",
    "skinl_derm",
    "skinl_photo",
]

backbones = [
    # "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    # "ViT-B-16,openai",
    # "ViT-L-14,openai",
    # "ViT-g-14,laion2b-s34b-b88k",
    "ViT-H-14,laion2b-s32b-b79k",
    "ViT-g-14,laion2b-s12b-b42k",
]


n_support = 1
n_query = 10
metatest_length = 100


if __name__ == "__main__":
    for dataset in datasets:
        for backbone in backbones:
            print(
                f"Running zero-shot test on dataset {dataset} with backbone {backbone}"
            )

            args = argparse.Namespace(
                prompt_dir="data/class_prompts",
                data_path="data/MedIMeta",
                presampled_data_path="data/MedIMeta_presampled",
                target_dataset=dataset,
                target_task=None,
                target_task_id=0,
                backbone=backbone,
                n_support=n_support,
                n_query=n_query,
                metatest_length=metatest_length,
                test_dir=test_dir,
            )

            zero_shot_test_he.main(args)
