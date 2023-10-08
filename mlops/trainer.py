import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from get_device import get_device
from tqdm import tqdm


class Trainer(object):
    def __init__(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        criterion: torch.nn.modules.loss,
        model: nn.Module,
        optimizer: torch.optim,
        shed: torch.optim.lr_scheduler,
        device=None,
        save_dir="../runs/",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = epoch
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.shed = shed
        self.device = device
        self.save_dir = save_dir

        if self.device is None:
            self.device = get_device()

        criterion_name = type(self.criterion).__name__
        model_name = type(self.model).__name__
        self.save_dir += f"//{criterion_name}_{model_name}//"
        self.save_dir = Path(self.save_dir)

        if not Path.exists(self.save_dir):
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        train_losses = []
        val_losses = []
        lrs = []
        min_loss = np.inf
        decrease = 1
        not_improve = 0

        self.model.to(self.device)
        fit_time = time.time()
        for e in range(self.epoch):
            since = time.time()

            # training loop
            self.model.train()
            tr_loss = self._train_step()

            # validation loop
            self.model.eval()
            val_loss = self._val_step()

            # calculatio mean for each batch
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            if min_loss > val_loss:
                print("LossDecreas.. {:.3f} >> {:.3f} ".format(min_loss, val_loss))
                min_loss = val_loss
                decrease += 1

                print("saving model...")
                self.save_model("best_model")

            else:
                not_improve += 1
                min_loss = val_loss
                print(f"Loss Not Decrease for {not_improve} time")

            print(
                "Epoch:{}/{}..".format(e + 1, self.epoch),
                "Train Loss: {:.3f}..".format(tr_loss),
                "Val Loss: {:.3f}..".format(val_loss),
                "Time: {:.2f}m".format((time.time() - since) / 60),
                "\n",
            )

            # step the learning rate
            lrs.append(self.get_lr())
            self.shed.step(val_losses[-1])

        self.save_model("last_model")

        history = {"train_loss": train_losses, "val_loss": val_losses, "lrs": lrs}
        print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
        return history

    def _train_step(self):
        running_loss = 0
        for _, data in enumerate(tqdm(self.train_loader)):
            # training phase
            image_tiles = data[0]

            image = image_tiles.to(self.device)
            output = self.model(image)
            loss = self.criterion(output, image)
            loss.backward()
            self.optimizer.step()  # update weight
            self.optimizer.zero_grad()  # reset gradient

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _val_step(self):
        val_loss = 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.val_loader)):
                image_tiles = data[0]
                image = image_tiles.to(self.device)
                output = self.model(image)
                loss = self.criterion(output, image)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def save_model(self, name):
        sample = torch.rand(1, 1, 32, 32).to(self.device)
        scripted_model = torch.jit.trace(
            self.model.to(self.device), sample.to(self.device)
        )
        scripted_model.save(f"{self.save_dir}/{name}.pth")
