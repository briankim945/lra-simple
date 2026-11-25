from setuptools import setup, find_packages
setup(
    name="lra-simple",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",  # torch.cuda.OutOfMemoryError
        "torchvision",
        "timm>=0.9",    # num_prefix_tokens / patch_embed.grid_size for center readout
        "numpy",
        "pandas",
        "pillow",
        "tqdm",
        "filelock",
    ],
)
