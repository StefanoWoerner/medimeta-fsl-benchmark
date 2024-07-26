from torch import nn
from transformers import (
    CLIPModel,
    ViTModel,
    AutoModel,
    ResNetModel,
)


class CLIPBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(backbone_name)

    def forward(self, inputs):
        return self.clip_model.get_image_features(inputs)


class ViTBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.vit_model = ViTModel.from_pretrained(backbone_name)

    def forward(self, inputs):
        return self.vit_model(inputs).pooler_output


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.model = ResNetModel.from_pretrained(backbone_name)

    def forward(self, images):
        return self.model(images).pooler_output.flatten(1)


class AutoBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(backbone_name)

    def forward(self, images):
        return self.model(images).pooler_output


def get_backbone_from_name(backbone_name):
    if "clip" in backbone_name:
        backbone = CLIPBackbone(backbone_name)
        backbone_num_features = backbone.clip_model.config.projection_dim
    elif "vit" in backbone_name:
        backbone = ViTBackbone(backbone_name)
        backbone_num_features = backbone.vit_model.config.hidden_size
    elif "resnet" in backbone_name:
        backbone = ResNetBackbone(backbone_name)
        backbone_num_features = backbone.model.config.hidden_sizes[-1]
    elif "/" in backbone_name:
        backbone = AutoBackbone(backbone_name)
        config = backbone.model.config
        if hasattr(config, "hidden_size"):
            backbone_num_features = config.hidden_size
        elif hasattr(config, "projection_dim"):
            backbone_num_features = config.projection_dim
        elif hasattr(config, "hidden_sizes"):
            backbone_num_features = config.hidden_sizes[-1]
        else:
            raise ValueError(f"Unknown backbone {backbone_name}")
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")

    return backbone, backbone_num_features
