# MedIMeta fine-tuning experiments
**Navigating Data Scarcity using Foundation Models: A Benchmark of
Few-Shot and Zero-Shot Learning Approaches in Medical Imaging**

This repository contains the code for the fine-tuning, linear probing
and zero-shot experiments of the paper "Navigating Data Scarcity using
Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning
Approaches in Medical Imaging".

The paper is available on [arXiv](https://arxiv.org/abs/2408.08058).

## Data
We use the MedIMeta dataset, which is a collection of 19 medical imaging
datasets spanning different modalities and tasks, all standardized to the
same format. It comes with a user-friendly Python package to directly load
images for use in PyTorch.

[Dataset website](https://www.woerner.eu/projects/medimeta/)

## How to replicate the experiments
1. Clone the repository
2. Install the requirements
3. Download the datasets and unpack them in the `data/MedIMeta` folder
4. Presample FSL tasks using the task presampling code contained in the
MedIMeta repository.
5. Optionally run a hyperparameter search using `run_all_find_hparams2.py`
6. If you chose to run the hyperparameter search, run the main experiments
with `deploy_all_ft_test_with_hparams.py`. Otherwise, you can run the main
experiments directly with `deploy_all_ft_test_fixedparams.py`, which uses
preset hyperparameters.
7. Run the zero-shot experiments with `run_all_zero_shot_test.py` and
`run_all_zero_shot_test_he.py`.

## Citation
If you use this code or our findings in your research, please cite the
following paper:

```
@misc{woerner2024navigatingdatascarcityusing,
      title={Navigating Data Scarcity using Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning Approaches in Medical Imaging}, 
      author={Stefano Woerner and Christian F. Baumgartner},
      year={2024},
      eprint={2408.08058},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.08058}, 
}
```

