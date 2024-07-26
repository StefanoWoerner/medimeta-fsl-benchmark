from typing import Literal

import torch
from lightning import pytorch as pl
from torchcross.cd.activations import get_prob_func
from torchcross.cd.losses import get_criterion
from torchcross.cd.metrics import Accuracy, AUROC
from torchcross.data import TaskDescription, TaskTarget
from torchmetrics import MetricCollection, Metric

# from transformers import CLIPModel, CLIPProcessor

from open_clip import (
    create_model_from_pretrained,
    get_tokenizer,
    create_model_and_transforms,
)  # works on open-clip-torch>=2.23.0, timm>=0.9.8


class CLIPClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        task_description: TaskDescription,
        prompt_template: str = None,
        class_prompts: list[str] = None,
        nothing_prompt_is_last: bool = False,
        pos_class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__()

        # self.model = CLIPModel.from_pretrained(backbone_name)
        # self.processor = CLIPProcessor.from_pretrained(backbone_name)

        if "," in backbone_name:
            backbone_name, checkpoint = backbone_name.split(",")
            self.model, self.transform = create_model_from_pretrained(
                backbone_name, pretrained=checkpoint
            )
        else:
            backbone_name = f"hf-hub:{backbone_name}"
            self.model, self.transform = create_model_from_pretrained(backbone_name)

        self.tokenizer = get_tokenizer(backbone_name)
        self.model.eval()

        self.task_description = task_description

        if prompt_template is None and class_prompts is None:
            raise ValueError("Either prompt_template or class_prompts must be provided")
        if prompt_template is not None and class_prompts is not None:
            raise ValueError(
                "Only one of prompt_template and class_prompts can be provided"
            )

        if prompt_template is not None:
            self.texts = [
                prompt_template.format(class_name=class_name)
                for class_name in task_description.classes.values()
            ]
            self.nothing_text = prompt_template.format(class_name="nothing")
        else:
            self.texts = class_prompts
            if nothing_prompt_is_last:
                self.nothing_text = self.texts[-1]
                self.texts = self.texts[:-1]
            else:
                self.nothing_text = "a photo of nothing"

        self.register_buffer(
            "tokens",
            self.tokenizer(self.texts),
        )
        self.register_buffer(
            "nothing_token",
            self.tokenizer(self.nothing_text),
        )

        self.register_buffer("pos_class_weights", pos_class_weights)
        self.criterion = get_criterion(
            task_description, pos_class_weights=pos_class_weights
        )
        self.pred_func = get_prob_func(task_description.task_target)

        self.training_metrics = MetricCollection(
            {"accuracy": Accuracy(task_description), "AUROC": AUROC(task_description)},
            postfix="/train",
        )
        self.validation_metrics = self.training_metrics.clone(postfix="/val")
        self.test_metrics = self.training_metrics.clone(postfix="/test")

        self.automatic_optimization = False

    # def forward(self, images: torch.Tensor) -> torch.Tensor:
    #     inputs = self.processor(
    #         text=self.texts,
    #         images=images,
    #         return_tensors="pt",
    #         padding=True,
    #     )
    #
    #     # put inputs on device
    #     inputs = {
    #         k: v.to(self.device) if isinstance(v, torch.Tensor) else v
    #         for k, v in inputs.items()
    #     }
    #
    #     outputs = self.model(**inputs)
    #     logits_per_image = (
    #         outputs.logits_per_image
    #     )  # this is the image-text similarity score
    #     return logits_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images = x
        texts = self.tokens
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(images, texts)
            logits = (logit_scale * image_features @ text_features.t()).detach()
            if self.task_description.task_target == TaskTarget.BINARY_CLASSIFICATION:
                logits = logits[:, 1:] - logits[:, :1]
            elif (
                self.task_description.task_target
                == TaskTarget.MULTILABEL_CLASSIFICATION
            ):
                image_features, text_features, logit_scale = self.model(
                    images, torch.cat((texts, self.nothing_token))
                )
                logits = (logit_scale * image_features @ text_features.t()).detach()
                logits = logits[:, :-1] - logits[:, -1:]
        return logits

    def do_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        pred = self.pred_func(logits)
        self.update_metrics(mode, pred, y)
        return loss

    def update_metrics(
        self, mode: Literal["train", "val", "test"], pred: torch.Tensor, y: torch.Tensor
    ):
        metrics = self.get_metrics(mode)
        return metrics(pred, y)

    def get_metrics(
        self, mode: Literal["train", "val", "test"]
    ) -> Metric | MetricCollection:
        if mode == "train":
            return self.training_metrics
        elif mode == "val":
            return self.validation_metrics
        elif mode == "test":
            return self.test_metrics
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss = self.do_step(batch, "val")

        self.log("loss/val", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.validation_metrics, prog_bar=True)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss = self.do_step(batch, "test")

        self.log("loss/test", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.test_metrics)
