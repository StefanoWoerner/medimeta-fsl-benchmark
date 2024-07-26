import os

test_dir = "./experiments/find_hparams2"

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

ns = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
n_query = 10
metatest_length = 10
n_trials = 50
linear_probe = True

for n_support in ns:
    for backbone in backbones:
        print(f"Deploying {backbone} hyperparameter search on {n_support}-shot tasks")
        command = (
            f"sbatch deploy_find_hparams2.sh"
            f" --data_path=data/MedIMeta --presampled_data_path=data/MedIMeta_presampled"
            f" --test_dir={test_dir} --n_trials={n_trials}"
            f" --backbone={backbone} --n_support={n_support} --n_query={n_query}"
            f" --metatest_length={metatest_length}"
            f"{' --linear_probe' if linear_probe else ''}"
        )
        os.system(command)
