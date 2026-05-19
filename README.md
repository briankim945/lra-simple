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