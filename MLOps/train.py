import os
import time

import numpy as np
import torch
import torch.nn as nn
from dataset import MLOpsDataset
from model import UNet
from torchvision import transforms
from tqdm import tqdm

TRAIN_PATH = "../data/train/"

DEVICE = "cpu"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_step(model, device, train_loader, criterion, optimizer):
    running_loss = 0
    for _, data in enumerate(tqdm(train_loader)):
        # training phase
        image_tiles = data[0]

        image = image_tiles.to(device)
        # forward
        output = model(image)
        loss = criterion(output, image)

        # backward
        loss.backward()
        optimizer.step()  # update weight
        optimizer.zero_grad()  # reset gradient

        running_loss += loss.item()

    return running_loss / len(train_loader)


def val_step(model, device, val_loader, criterion):
    val_loss = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader)):
            image_tiles = data[0]
            image = image_tiles.to(device)

            output = model(image)
            # loss
            loss = criterion(output, image)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def save_model(model, save_path, name):
    sample = torch.rand(1, 1, 32, 32).to(DEVICE)
    scripted_model = torch.jit.trace(model.to(DEVICE), sample.to(DEVICE))
    scripted_model.save(f"{save_path}/{name}.pth")


def train(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    save_path="../runs/",
):
    train_losses = []
    val_losses = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    criterion_name = type(criterion).__name__
    model_name = type(model).__name__
    save_dir = f"{save_path}/{criterion_name}" f"_{model_name}/"

    model.to(DEVICE)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()

        # training loop
        model.train()
        tr_loss = fit_step(model, DEVICE, train_loader, criterion, optimizer)

        # validation loop
        model.eval()
        val_loss = val_step(model, DEVICE, val_loader, criterion)

        # calculatio mean for each batch
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if min_loss > val_loss:
            print("LossDecreas.. {:.3f} >> {:.3f} ".format(min_loss, val_loss))
            min_loss = val_loss
            decrease += 1

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("saving model...")
            save_model(model, save_dir, "best_model")

        else:
            not_improve += 1
            min_loss = val_loss
            print(f"Loss Not Decrease for {not_improve} time")

        print(
            "Epoch:{}/{}..".format(e + 1, epochs),
            "Train Loss: {:.3f}..".format(tr_loss),
            "Val Loss: {:.3f}..".format(val_loss),
            "Time: {:.2f}m".format((time.time() - since) / 60),
        )

        # step the learning rate
        lrs.append(get_lr(optimizer))
        scheduler.step(val_losses[-1])

    save_model(model, save_dir, "last_model")

    history = {"train_loss": train_losses, "val_loss": val_losses, "lrs": lrs}
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history


if __name__ == "__main__":
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    tr_dataset = MLOpsDataset(TRAIN_PATH, 40, transform_train)
    tr_dataset, val_dataset = torch.utils.data.random_split(
        tr_dataset, [int(len(tr_dataset) * 0.8), int(len(tr_dataset) * 0.2)]
    )

    train_dataloader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=16, shuffle=True, num_workers=0
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=True, num_workers=0
    )

    print(
        f" size train_dataset: {len(tr_dataset)}"
        f"\n size val_dataset: {len(val_dataset)}\n"
    )

    max_lr = 1e-2
    epoch = 40
    weight_decay = 1e-4

    criterion = nn.L1Loss()
    model = UNet(in_chan=1, n_clas=1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = train(
        epoch,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        sched,
        save_path="../runs/",
    )
