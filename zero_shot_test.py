import os
from itertools import repeat

import lightning.pytorch as pl
import pandas as pd
import torchcross as tx
from lightning.pytorch.loggers import CSVLogger
from medimeta import MedIMeta, PickledMedIMetaTaskDataset
from torchvision import transforms

from models.clip_classifier import CLIPClassifier


def main(args):
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
        n_support=1,
        n_query=args.n_query,
        length=args.metatest_length,
        collate_fn=tx.utils.collate_fn.stack,
        transform=transforms.Compose([]),  # transforms.Compose(standard_transforms),
    )
    metatest_dataset.task_source.num_channels = 3

    # prompt_prefix = f"A {metatest_dataset.task_source.dataset_name} image of a"
    # prompt_prefix = "A medical image of a"
    # prompt_prefix = "A photo of a"
    prompt_kwargs = {
        "dataset_name": metatest_dataset.task_source.dataset_name,
        "task_name": metatest_dataset.task_source.task_name,
        "domain_identifier": metatest_dataset.task_source.domain_identifier,
        "task_identifier": metatest_dataset.task_source.task_identifier,
        "dataset_id": target_dataset_id,
        "dataset_summary": metatest_dataset.task_source.info["summary"],
    }
    prompt_template = args.prompt_template.format(**prompt_kwargs)

    # Create unique experiment name and logger
    save_dir = f"{args.test_dir}/{target_dataset_id}_{target_task_name}/zero-shot"
    experiment_name = f"{args.backbone.replace('/', '__')}/{args.prompt_template}"

    csv_logger = CSVLogger(save_dir, name=experiment_name, flush_logs_every_n_steps=1)

    model = CLIPClassifier(
        args.backbone,
        metatest_dataset.task_source.task_description,
        prompt_template,
    )
    metatest_dataset.task_source.transform = model.transform

    all_test_metrics = []

    for task in metatest_dataset:
        # Create a new trainer for each task with the max_steps parameter
        trainer = pl.Trainer(
            max_steps=0,
            logger=False,  # [csv_logger],
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

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

    return mean["AUROC/test"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MedIMeta")
    parser.add_argument("--target_dataset", type=str, default="cxr")
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--target_task_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument(
        "--presampled_data_path", type=str, default="data/MedIMeta_presampled"
    )
    # parser.add_argument("--backbone", type=str, default="google/vit-base-patch16-224")
    # parser.add_argument("--backbone", type=str, default="openai/clip-vit-base-patch16")
    # parser.add_argument("--backbone", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--backbone",
        type=str,
        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    # parser.add_argument(
    #     "--backbone", type=str, default="ikim-uk-essen/BiomedCLIP_ViT_patch16_224"
    # )
    # parser.add_argument("--backbone", type=str, default="ikim-uk-essen/BiomedCLIP_ViT_patch16_224")
    # parser.add_argument("--backbone", type=str, default="microsoft/resnet-18")
    # parser.add_argument("--backbone", type=str, default="microsoft/resnet-50")
    parser.add_argument("--test_dir", type=str, default="./experiments/zero-shot_test")
    parser.add_argument("--n_support", type=int, default=1)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--metatest_length", type=int, default=100)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="{{class_name}}",
    )

    main(parser.parse_args())
