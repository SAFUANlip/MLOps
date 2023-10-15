import cv2
import pandas as pd
import torch
import torch.nn as nn
from mlops import cls_labels
from mlops.get_device import get_device
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

SAVE_PATH = "infer_result/"
model_path = "runs/L1Loss_UNet/"


def infer_step(model, device, test_loader, criterion, save=False):
    df = pd.DataFrame(columns=["img_name", "L1loss_recovery"])
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            image_tiles, label = data
            image = image_tiles.to(device)
            label = str(int(label))

            output = model(image)
            if save:
                output_img = output.squeeze().cpu().detach().numpy()
                cv2.imwrite(SAVE_PATH + str(i) + "_" + label + ".png", output_img * 255)

            # loss
            loss = criterion(output, image)
            val_loss += loss.item()
            df.loc[-1] = [label, loss.item()]  # adding a row
            df.index = df.index + 1  # shifting index
            df = df.sort_index()
    df.to_csv(SAVE_PATH + "results.csv")
    print(
        f" mean L1 loss {round(val_loss/len(test_loader),5)},\n"
        f" Images saved into {SAVE_PATH}"
    )


def inference():
    model = torch.jit.load(model_path + "best_model.pth")

    transforms_set = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32, 32), antialias=None)]
    )

    data = MNIST(root="../data", train=False, download=True, transform=transforms_set)

    idx = [cls in cls_labels for cls in data.targets]
    data.targets = data.targets[idx]
    data.data = data.data[idx]

    loss = nn.L1Loss()
    test_dataloader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0
    )

    device = get_device()

    infer_step(model, device, test_dataloader, loss, save=True)


if __name__ == "__main__":
    inference()
