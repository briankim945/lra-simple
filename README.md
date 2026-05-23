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

Running the tasks is controlled by two scripts: `launch_linear_probe.sh` and  `launch_finetune.sh`. Both scripts require a `--task` flag for the task being evaluated and a `--data_dir` pointing to the dataset being evaluated on. Further optional flags for running the pipeline, including the number of GPUs and the output directory, are provided in the scripts.

To modify the configurations used in the grid search, modify `lra_simple/configs.py`.

### Sample Tasks

- Pathfinder: a task for tracing a linear path between two dots on an image [Learning long-range spatial dependencies with horizontal gated recurrent units](https://proceedings.neurips.cc/paper/2018/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html)
- Cluttered ABC: judge whether two markers fall on the same or differ-
ent letters [Disentangling neural mechanisms for perceptual grouping](https://openreview.net/forum?id=HJxrVA4FDS)
- Parametric Synthetic Visual Reasoning Test: comparing image artifacts in separate frames [Not-So-CLEVR: learning same–different relations strains feedforward neural networks](https://royalsocietypublishing.org/rsfs/article/8/4/20180011/64218/Not-So-CLEVR-learning-same-different-relations)

### Example Commands

To run linear probing on Pathfinder:

```./launch_linear_probe.sh --task pathfinder --data_dir /path/to/data```

To run linear probing on Pathfinder with 4 GPUs:

```./launch_linear_probe.sh --task pathfinder --data_dir /path/to/data --gpus 4```

To run linear probing on cABC and a specified list of models in a text file:

```./launch_linear_probe.sh --task cabc --data_dir /path/to/data --models_list models.txt```

To run fine-tuning on cABC with a reduced batch size and gradient checkpointing:

```./launch_finetune.sh --task cabc --data_dir /path/to/data --gpus 4 --small_batch --gradient_checkpointing```
