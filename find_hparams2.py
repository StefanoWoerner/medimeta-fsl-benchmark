import logging
import math
import os

from finetune_and_test import ft_and_test

import optuna

import argparse

from optuna.storages import JournalStorage, JournalFileStorage


def main(args):
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

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

    def objective(trial):
        use_data_augmentation = trial.suggest_categorical(
            "use_data_augmentation", [False]
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        finetuning_steps = trial.suggest_int("finetuning_steps", 5, 200)
        optimizer = trial.suggest_categorical("optimizer", ["Adam"])
        head_init = trial.suggest_categorical("head_init", ["kaiming", "zero"])
        # linear_probe = trial.suggest_categorical("linear_probe", [False])

        trial.set_user_attr("linear_probe", args.linear_probe)

        print()
        print("-----------------------------------")
        print(f"Testing model {args.backbone}")
        print(f"Support shots: {args.n_support}")
        print(f"Query shots: {args.n_query}")
        print(f"Length: {args.metatest_length}")
        print("-----------------------------------")
        print(f"Hyperparameters:")
        print(f"Use data augmentation: {use_data_augmentation}")
        print(f"Learning rate: {learning_rate}")
        print(f"Finetuning steps: {finetuning_steps}")
        print(f"Optimizer: {optimizer}")
        print(f"Head init: {head_init}")
        print(f"Linear probe: {args.linear_probe}")
        print("-----------------------------------")

        means = []
        for dataset in datasets:
            print()
            print("-----------------------------------")
            print(f"Testing dataset {dataset}")
            print("-----------------------------------")

            args_to_use = argparse.Namespace(
                data_path=args.data_path,
                presampled_data_path=args.presampled_data_path,
                target_dataset=dataset,
                target_task=None,
                target_task_id=0,
                use_data_augmentation=use_data_augmentation,
                learning_rate=learning_rate,
                finetuning_steps=finetuning_steps,
                optimizer=optimizer,
                head_init=head_init,
                backbone=args.backbone,
                linear_probe=args.linear_probe,
                n_support=args.n_support,
                n_query=args.n_query,
                metatest_length=args.metatest_length,
                test_dir=args.test_dir,
            )

            mean = ft_and_test(args_to_use, write_csvs=False)
            means.append(mean[0])
        trial.set_user_attr(
            "geometric_average_AUROC", math.prod(means) ** (1 / len(means))
        )
        return means

    study_name = (
        f"{args.n_support}-shot/"
        f"{args.backbone.replace('/', '__')}__linear_probe={args.linear_probe}"
    )

    os.makedirs(os.path.dirname(f"{args.test_dir}/{study_name}.log"), exist_ok=True)

    storage = JournalStorage(JournalFileStorage(f"{args.test_dir}/{study_name}.log"))
    study = optuna.create_study(
        directions=["maximize"] * len(datasets),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    study.trials_dataframe().to_csv(f"{args.test_dir}/{study_name}.csv")
    study.trials_dataframe().to_pickle(f"{args.test_dir}/{study_name}.pkl")

    print("Finished study ", study_name)
    print("Number of finished trials: ", len(study.trials))
    print("Best trials:")
    trials = study.best_trials
    print(trials)
    print("-----------------------------------")

    # print("Best trial:")
    # trial = study.best_trial
    #
    # print("  Value: ", trial.value)
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    #
    # print("  User attrs:")
    # for key, value in trial.user_attrs.items():
    #     print("    {}: {}".format(key, value))
    #
    # print("  Intermediate values: ")
    # for key, value in trial.intermediate_values.items():
    #     print("    {}: {}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument(
        "--presampled_data_path", type=str, default="data/MedIMeta_presampled"
    )
    parser.add_argument("--backbone", type=str, default="microsoft/resnet-18")
    # parser.add_argument("--backbone", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument("--n_support", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--metatest_length", type=int, default=10)
    parser.add_argument("--test_dir", type=str, default="./experiments/find_hparams2")
    parser.add_argument("--n_trials", type=int, default=10)

    main(parser.parse_args())
