import torch
from mlops import cls_labels
from torchvision import transforms
from torchvision.datasets import MNIST


def get_fit_loaders():
    transforms_set = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32, 32), antialias=None)]
    )

    data = MNIST(root="../data", train=True, download=True, transform=transforms_set)

    idx = [cls in cls_labels for cls in data.targets]
    data.targets = data.targets[idx]
    data.data = data.data[idx]

    train_dataset, val_dataset = torch.utils.data.random_split(
        data, [int(len(data) * 0.85), int(len(data) * 0.15)]
    )

    print(
        f" size train_dataset: {len(train_dataset)}"
        f"\n size val_dataset: {len(val_dataset)}\n"
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=True, num_workers=0
    )
    return train_dataloader, val_dataloader
