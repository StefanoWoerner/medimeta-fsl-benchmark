# MedIMeta fine-tuning experiments
**Navigating Data Scarcity using Foundation Models: A Benchmark of
Few-Shot and Zero-Shot Learning Approaches in Medical Imaging**

This repository contains the code for the fine-tuning, linear probing
and zero-shot experiments of the paper "Navigating Data Scarcity using
Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning
Approaches in Medical Imaging". The paper is available on arXiv:
[TBD](https://arxiv.org/).

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
