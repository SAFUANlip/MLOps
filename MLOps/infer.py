import cv2
import pandas as pd
import torch
import torch.nn as nn
from dataset import MLOpsDataset
from torchvision import transforms
from tqdm import tqdm

TEST_PATH = "../data/test/"
SAVE_PATH = "../infer_result/"
DEVICE = "cpu"


def infer_step(model, device, test_loader, criterion, save=False):
    df = pd.DataFrame(columns=["img_name", "L1loss_recovery"])
    val_loss = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            image_tiles, image_name = data
            image = image_tiles.to(device)

            output = model(image)
            if save:
                output_img = output.squeeze().cpu().detach().numpy()
                cv2.imwrite(SAVE_PATH + image_name[0], output_img * 255)

            # loss
            loss = criterion(output, image)
            val_loss += loss.item()
            df.loc[-1] = [image_name[0], loss.item()]  # adding a row
            df.index = df.index + 1  # shifting index
            df = df.sort_index()
    df.to_csv(SAVE_PATH + "results.csv")
    print(
        f" mean L1 loss {round(val_loss/len(test_dataloader),5)},\n"
        f" Images saved into {SAVE_PATH}"
    )


if __name__ == "__main__":
    model_path = "../runs/L1Loss_UNet/"
    model = torch.jit.load(model_path + "best_model.pth")

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = MLOpsDataset(TEST_PATH, 10, transform_train)

    loss = nn.L1Loss()
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    infer_step(model, DEVICE, test_dataloader, loss, save=True)
