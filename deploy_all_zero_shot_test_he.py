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
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "ViT-B-16,openai",
    "ViT-L-14,openai",
    "ViT-H-14,laion2b-s32b-b79k",
    "ViT-g-14,laion2b-s12b-b42k",
]


if __name__ == "__main__":
    for dataset in datasets:
        for backbone in backbones:
            print(
                f"Deploying zero-shot test on dataset {dataset} with backbone {backbone}"
            )

            command = (
                f"sbatch deploy_zero_shot_test_he.sh "
                f"--prompt_dir=data/class_prompts "
                f"--data_path=data/MedIMeta "
                f"--presampled_data_path=data/MedIMeta_presampled "
                f"--target_dataset={dataset} "
                f"--backbone={backbone} "
                f"--test_dir={test_dir}"
            )

            os.system(command)
