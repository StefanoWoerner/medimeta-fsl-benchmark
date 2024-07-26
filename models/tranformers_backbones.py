from torch import nn
from transformers import (
    AutoModelForImageClassification,
    ViTForImageClassification,
    ResNetForImageClassification,
    Dinov2ForImageClassification,
)
from transformers.models.clip.modeling_clip import CLIPForImageClassification


class TransformersBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            backbone_name, num_labels=0
        )

    def forward(self, images):
        return self.model(images).logits


def get_backbone_from_name(backbone_name):
    backbone = TransformersBackbone(backbone_name)
    model = backbone.model

    if isinstance(model, CLIPForImageClassification):
        backbone_num_features = model.config.vision_config.hidden_size
    elif isinstance(model, ViTForImageClassification):
        backbone_num_features = model.config.hidden_size
    elif isinstance(model, ResNetForImageClassification):
        backbone_num_features = model.config.hidden_sizes[-1]
    elif isinstance(model, Dinov2ForImageClassification):
        backbone_num_features = model.config.hidden_size * 2
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")
    return backbone, backbone_num_features
