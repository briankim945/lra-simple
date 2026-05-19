# Long-Range Reasoning Pipeline (simplified)

This repo provides the framework for examining the performance of a library of pre-trained PyTorch models on a set of long-range visual reasoning tasks. Evaluation can be done either using linear probing on a frozen set of model weights or fine-tuning an existing model on a reasoning task.

## Setup

```bash
git clone https://github.com/briankim945/lra-three-zoo.git
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