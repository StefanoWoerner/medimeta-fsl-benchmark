import os

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import pandas as pd

hparam_search_dir = "./experiments/find_hparams2"

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

best_arithmetic_values = []
best_geometric_values = []
best_harmonic_values = []
for backbone in backbones:
    for n_support in n_supports:
        study_name = f"{n_support}-shot/{backbone.replace('/', '__')}__linear_probe={linear_probe}"
        storage = JournalStorage(
            JournalFileStorage(f"{hparam_search_dir}/{study_name}.log")
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        if study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
            trials_df = study.trials_dataframe()
            value_columns = [c for c in trials_df.columns if c.startswith("value")]
            # replace nan with epsilon
            trials_df[value_columns] = trials_df[value_columns].fillna(1e-6)

            # # print trial index and study name if negative values
            # for i, row in trials_df.iterrows():
            #     if row[value_columns].min() <= 0:
            #         print(i, study_name)
            # trials_df[value_columns].apply(lambda x: x.apply(lambda v: v if v > 0 else print(v, i, study_name)), axis=1)
            # get arithmetic mean
            arithmetic_average = trials_df[value_columns].mean(axis=1)
            # get geometric mean
            geometric_average = trials_df[value_columns].apply(
                lambda x: x.apply(lambda v: v if v > 0 else 0).prod() ** (1 / len(x)),
                axis=1,
            )
            # get harmonic mean
            harmonic_average = trials_df[value_columns].apply(
                lambda x: len(x) / sum(1 / v for v in x), axis=1
            )

            # get maximum values
            best_arithmetic = arithmetic_average.max()
            best_harmonic = harmonic_average.max()
            best_geometric = geometric_average.max()

            # get argmax
            best_arithmetic_idx = arithmetic_average.idxmax()
            best_harmonic_idx = harmonic_average.idxmax()
            best_geometric_idx = geometric_average.idxmax()

            # collect all objective values for argmax
            best_arithmetic_values.append(
                {
                    "backbone": backbone,
                    "n_support": n_support,
                    "arithmetic average AUROC": best_arithmetic,
                    "trial": best_arithmetic_idx,
                }
                | trials_df.loc[best_arithmetic_idx, :].to_dict()
            )
            best_geometric_values.append(
                {
                    "backbone": backbone,
                    "n_support": n_support,
                    "geometric average AUROC": best_geometric,
                    "trial": best_geometric_idx,
                }
                | trials_df.loc[best_geometric_idx, :].to_dict()
            )
            best_harmonic_values.append(
                {
                    "backbone": backbone,
                    "n_support": n_support,
                    "harmonic average AUROC": best_harmonic,
                    "trial": best_harmonic_idx,
                }
                | trials_df.loc[best_harmonic_idx, :].to_dict()
            )

df_arithmetic = pd.DataFrame(best_arithmetic_values)
df_geometric = pd.DataFrame(best_geometric_values)
df_harmonic = pd.DataFrame(best_harmonic_values)


test_dir = "./experiments/finetune_and_test2_g2"

# deploy for each df row
for i, row in df_harmonic.iterrows():
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

print(f"Length of df_harmonic: {len(df_harmonic)}")
print(f"Length of dataset_ids: {len(dataset_ids)}")
print(f"Total number of jobs: {len(df_harmonic) * len(dataset_ids)}")
