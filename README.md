# MLOps
Homeetask for MLOps course

Train UNet model to restore Mnist Dataset, now only 1 and 6 \
The idea use it like autoencoder to find anomalies on images

Some instructions:

1. Download repository
2. You should have installed `poetry`
3. In cmd in downloaded directory run: `poetry install`
4. To use created env run: `poetry shell`
5. To train model: `python train.py`
6. To infer model: `python infer.py`

Addition:

Also, you can run pre-commit and check standards

1. Install pre-commit by `pip install pre-commit`
2. Check standards: `pre-commit run -a`

If you use PyCharm and run `poetry-install`, then env will be installed in some hole
and easier will be compy this env from hole to your current directory (I am sure,
that there are some better solution, but I haven't found it yet)
