import os
from functools import partial
from itertools import repeat

import lightning.pytorch as pl
import pandas as pd
import torch
import torchcross as tx
from lightning.pytorch.loggers import CSVLogger
from medimeta import MedIMeta, PickledMedIMetaTaskDataset
from torchcross.models.lightning import (
    SimpleClassifier,
)
from torchvision import transforms

from models.tranformers_backbones import get_backbone_from_name


def main(args):
    ns = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

    if args.n_support is not None:
        ns = [args.n_support]

    for n in ns:
        args.n_support = n
        args.n_query = 10
        args.metatest_length = 100

        print()
        print("-----------------------------------")
        print(f"Testing model {args.backbone}")
        print(f"Support shots: {args.n_support}")
        print(f"Query shots: {args.n_query}")
        print(f"Length: {args.metatest_length}")
        print("-----------------------------------")
        ft_and_test(args)
        print("-----------------------------------")


def ft_and_test(args, write_csvs=True):
    data_path = args.data_path
    presampled_data_path = args.presampled_data_path
    target_dataset_id = args.target_dataset
    target_task_name = args.target_task
    target_task_id = args.target_task_id

    if target_task_name and target_task_id:
        raise ValueError(
            "Only one of target_task_name and target_task_id can be specified"
        )
    elif target_task_name is None:
        dataset_info = MedIMeta.get_info_dict(data_path, target_dataset_id)
        target_task_name = dataset_info["tasks"][target_task_id]["task_name"]

    augmentation_transforms = []
    if args.use_data_augmentation:
        random_crop_and_resize = transforms.Compose(
            [
                transforms.RandomCrop(200),
                transforms.Resize(224),
            ]
        )

        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(
                10, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomApply([random_crop_and_resize], p=0.25),
        ]
    standard_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Create the test dataloader
    metatest_dataset = PickledMedIMetaTaskDataset(
        presampled_data_path,
        data_path,
        target_dataset_id,
        target_task_name,
        n_support=args.n_support,
        n_query=args.n_query,
        length=args.metatest_length,
        collate_fn=tx.utils.collate_fn.stack,
        transform=transforms.Compose(augmentation_transforms + standard_transforms),
    )
    metatest_dataset.task_source.num_channels = 3

    hparams = {
        "lr": args.learning_rate,
    }

    # Create unique experiment name and logger
    save_dir = (
        f"{args.test_dir}/{target_dataset_id}_{target_task_name}/"
        f"{args.n_support}-shot"
    )
    experiment_name = (
        f"{args.backbone.replace('/', '__')}/optim={args.optimizer}_lr={args.learning_rate}"
        f"_steps={args.finetuning_steps}_hinit={args.head_init}_linear={args.linear_probe}_aug={args.use_data_augmentation}"
    )

    csv_logger = CSVLogger(save_dir, name=experiment_name, flush_logs_every_n_steps=1)

    # Create optimizer
    if args.optimizer == "Adam":
        optimizer = partial(torch.optim.Adam, **hparams)
    elif args.optimizer == "SGD":
        optimizer = partial(torch.optim.SGD, **hparams)
    else:
        raise ValueError("Unknown optimizer")

    all_test_metrics = []

    if args.linear_probe:
        backbone, num_backbone_features = get_backbone_from_name(args.backbone)

    for task in metatest_dataset:
        if not args.linear_probe:
            backbone, num_backbone_features = get_backbone_from_name(args.backbone)

        model = SimpleClassifier(
            torch.nn.Identity() if args.linear_probe else backbone,
            num_backbone_features,
            task.description,
            optimizer,
            expand_input_channels=False,
            head_init=args.head_init,
            linear_probe=args.linear_probe,
        )

        # Create a new trainer for each task with the max_steps parameter
        trainer = pl.Trainer(
            max_steps=args.finetuning_steps,
            logger=False,  # [csv_logger],
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        if args.linear_probe:
            backbone = backbone.to(trainer.device_ids[0])

            with torch.no_grad():
                support_features = backbone(task.support[0].to(trainer.device_ids[0]))
                query_features = backbone(task.query[0].to(trainer.device_ids[0]))

            # Fine-tune the model on the current task
            trainer.fit(
                model,
                repeat((support_features, task.support[1])),
                repeat((query_features, task.query[1]), 1),
            )

            test_metrics = trainer.test(
                model,
                repeat((query_features, task.query[1]), 1),
            )
        else:
            # Fine-tune the model on the current task
            trainer.fit(model, repeat(task.support), repeat(task.query, 1))

            test_metrics = trainer.test(model, repeat(task.query, 1))
        all_test_metrics.append(test_metrics[0])

    if write_csvs:
        os.makedirs(csv_logger.log_dir, exist_ok=True)

    print("All Test Metrics:", all_test_metrics)
    df = pd.DataFrame(all_test_metrics)
    if write_csvs:
        df.to_csv(f"{csv_logger.log_dir}_test_metrics.csv")

    # compute mean and std and count and save them to a csv with one row per mean/std/count
    mean = df.mean()
    std = df.std()
    count = df.count()
    agg_df = pd.concat([mean, std, count], axis=1)
    agg_df.columns = ["mean", "std", "count"]
    if write_csvs:
        agg_df.to_csv(f"{csv_logger.log_dir}_test_metrics_aggregated.csv")

    return mean["AUROC/test"], mean["accuracy/test"], mean["loss/test"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="oct")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--finetuning_steps", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--head_init", type=str, default="zero")
    parser.add_argument(
        "--presampled_data_path", type=str, default="data/MedIMeta_presampled"
    )
    parser.add_argument("--backbone", type=str, default="google/vit-base-patch16-224")
    # parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    # parser.add_argument("--backbone", type=str, default="ikim-uk-essen/BiomedCLIP_ViT_patch16_224")
    # parser.add_argument("--backbone", type=str, default="microsoft/resnet-18")
    # parser.add_argument("--backbone", type=str, default="microsoft/resnet-50")
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument(
        "--test_dir", type=str, default="./experiments/finetune_and_test"
    )
    parser.add_argument("--n_support", type=int, default=None)

    main(parser.parse_args())
