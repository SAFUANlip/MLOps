import torch
import torch.nn as nn
from mlops.dataset import get_fit_loaders
from mlops.models_architecture.UNet import UNet
from mlops.trainer import Trainer


def train():
    train_loader, val_loader = get_fit_loaders()

    epoch = 5
    criterion = nn.L1Loss()
    model = UNet(in_chan=1, n_clas=1)

    max_lr = 1e-2
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    shed = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    tr_unet = Trainer(
        train_loader,
        val_loader,
        epoch=epoch,
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        shed=shed,
    )

    tr_unet.train()


if __name__ == "__main__":
    train()
