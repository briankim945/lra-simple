# Long-Range Reasoning Pipeline (simplified)

This repo provides a framework for evaluating pre-trained PyTorch vision models on a set of long-range visual reasoning tasks. It supports systematic evaluation of hundreds to thousands of models across multiple tasks via grid search distributed across multiple GPUs, with automated logging, result aggregation, and pipeline pausing/resuming. Evaluation can be done either using linear probing on a frozen set of model weights or applying fine-tuning to an existing model on a reasoning task.

## Setup

```bash
git clone https://github.com/briankim945/lra-simple.git
cd lra-simple
pip install -e .
```

To remove, run:

```pip uninstall lra-simple```

## Structure

### Linear Probing

The pipeline takes in a set of input data and produces the frozen weights of the target model at the penultimate depth for each image. We instantiate a range of lightweight linear probes using a predefined set of configurations for a grid search. The linear probes are trained and evaluated on the model's frozen weights, with the top-performing being recorded and its weights and configuration being saved.

### Fine-Tuning

The pipeline uses the trainable attribute of PyTorch models to further fine-tune a given model on the provided dataset. This is significantly more compute- and time-intensive than linear probing.

## Usage

Running the tasks is controlled by two scripts in `scripts/`: `launch_linear_probe.sh` and `launch_finetune.sh`, run from the repo root. Both scripts require a `--task` flag for the task being evaluated, a `--data_dir` pointing to the dataset being evaluated on, and the models to evaluate: either `--models_list` (a text file with one timm model name per line) or `--models_csv` (a CSV with a `model_name` column). Further optional flags for running the pipeline, including the number of GPUs and the output directory, are listed with `--help`.

To modify the configurations used in the grid search, modify `lra_simple/configs.py`. For a quick fine-tuning run (e.g. a smoke test), `--reduced_grid` uses a 6-configuration grid instead of the full one.

### Sample Tasks

- Pathfinder: a task for tracing a linear path between two dots on an image [Learning long-range spatial dependencies with horizontal gated recurrent units](https://proceedings.neurips.cc/paper/2018/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html)
- Cluttered ABC: judge whether two markers fall on the same or different letters [Disentangling neural mechanisms for perceptual grouping](https://openreview.net/forum?id=HJxrVA4FDS)
- Parametric Synthetic Visual Reasoning Test: comparing image artifacts in separate frames [Not-So-CLEVR: learning same–different relations strains feedforward neural networks](https://royalsocietypublishing.org/rsfs/article/8/4/20180011/64218/Not-So-CLEVR-learning-same-different-relations)
- Tilt: regressing the orientation of a center grating in a center-surround stimulus. Unlike the other tasks, the model is read out at the center of the image (the center token for ViTs, the center position of the final feature map for CNNs), and results are reported as mean absolute angular error in degrees (lower is better) rather than accuracy. These errors are stored under the same `best_val_acc` / `test_acc` keys as the other tasks' accuracies.

New tasks are registered in `TASK_CONFIGS` (`lra_simple/configs.py`) with a dataset loader in `DATASET_LOADERS` (`lra_simple/common.py`).

### Example Commands

To run linear probing on Pathfinder for the models listed in `models.txt`:

```./scripts/launch_linear_probe.sh --task pathfinder --data_dir /path/to/data --models_list models.txt```

To run linear probing on Pathfinder with 4 GPUs:

```./scripts/launch_linear_probe.sh --task pathfinder --data_dir /path/to/data --models_list models.txt --gpus 4```

To run linear probing on cABC:

```./scripts/launch_linear_probe.sh --task cabc --data_dir /path/to/data --models_list models.txt```

To run fine-tuning on cABC with a reduced batch size and gradient checkpointing:

```./scripts/launch_finetune.sh --task cabc --data_dir /path/to/data --models_list models.txt --gpus 4 --small_batch --gradient_checkpointing```

To run linear probing on Tilt (`--data_dir` contains `train/` and `test/`):

```./scripts/launch_linear_probe.sh --task tilt --data_dir /path/to/data --models_list models.txt```
