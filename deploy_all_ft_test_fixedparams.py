import os

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import pandas as pd

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
    # "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    # "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
]

dataset_ids = [
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

n_supports = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

linear_probe = True


fixed_params = {
    "params_learning_rate": 1e-4,
    "params_finetuning_steps": 150,
    "params_optimizer": "Adam",
    "params_head_init": "zero",
    "user_attrs_linear_probe": linear_probe,
}
fixed_param_records = []
for backbone in backbones:
    for n_support in n_supports:
        fixed_param_records.append(
            {"backbone": backbone, "n_support": n_support, **fixed_params}
        )

fixed_params_df = pd.DataFrame(fixed_param_records)

test_dir = "./experiments/finetune_and_test_fixedparams"

# deploy for each df row
for i, row in fixed_params_df.iterrows():
    for target_dataset in dataset_ids:
        print(
            f"Deploying model {row['backbone']} with hyperparameters "
            f"learning_rate={row['params_learning_rate']} "
            f"finetuning_steps={row['params_finetuning_steps']} "
            f"optimizer={row['params_optimizer']} "
            f"head_init={row['params_head_init']} "
            f"on {row['n_support']}-shot {target_dataset} tasks"
        )
        command = (
            f"sbatch deploy_ft_test.sh "
            f"--data_path=data/MedIMeta "
            f"--presampled_data_path=data/MedIMeta_presampled "
            f"--target_dataset={target_dataset} "
            f"--backbone={row['backbone']} "
            f"--learning_rate={row['params_learning_rate']} "
            f"--finetuning_steps={row['params_finetuning_steps']} "
            f"--optimizer={row['params_optimizer']} "
            f"--head_init={row['params_head_init']} "
            f"--n_support={row['n_support']} "
            f"--test_dir={test_dir}"
        )
        if row["user_attrs_linear_probe"]:
            command += " --linear_probe"
        # submit job
        os.system(command)
        # print(command)

print(f"Length of df_harmonic: {len(fixed_params_df)}")
print(f"Length of dataset_ids: {len(dataset_ids)}")
print(f"Total number of jobs: {len(fixed_params_df) * len(dataset_ids)}")
