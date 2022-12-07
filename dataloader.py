import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class KDD22(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, val_split=0.2, transform=None, target_transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # read csv and split train and test
        data = pd.read_csv(os.path.join(data_dir, "public.csv"), sep=",")
        data_train = data[~data['North'].isna()].reset_index(drop=True)
        data_test = data[data['North'].isna()].reset_index(drop=True)

        # split train and val (10% of training data)
        idx_val = round((1 - val_split) * len(data_train))
        data_val = data_train.iloc[idx_val:, :].reset_index(drop=True)
        data_train = data_train.iloc[:idx_val, :]

        if split == "train":
            self.data = data_train
        elif split == "test":
            self.data = data_test
        elif split == "val":
            self.data = data_val
            split = "train"
        self.img_dir = os.path.join(data_dir, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.filename = self.data.loc[idx, "Filename"]
        img_path = os.path.join(self.img_dir, self.filename)
        img = Image.open(img_path).convert('L')
        img = np.asarray(img)

        h, w = img.shape
        img1 = np.copy(img[:, :w // 2])  # previous frame
        img2 = np.copy(img[:, w // 2:])  # current frame

        altitude = self.data.loc[idx, "Altitude"]
        delta = self.data.loc[idx, "Delta"]
        odom = self.data.iloc[idx, 3:].values  # [north, east]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.target_transform:
            odom = self.target_transform(odom)

        # concatenate in first axis
        images = np.concatenate([img1, img2], axis=0)
        images = np.asarray(images)

        # normalize and convert to tensor
        alt_mean = 187.513774
        alt_std = 18.851626
        altitude = (altitude - alt_mean) / alt_std
        altitude = np.asarray(altitude)
        altitude = torch.from_numpy(altitude)

        delta_mean = -0.022164
        delta_std = 0.224125
        delta = (delta - delta_mean) / delta_std
        delta = np.asarray(delta)
        delta = torch.from_numpy(delta)

        return images, altitude, delta, np.asarray(odom, dtype="float64")


if __name__ == "__main__":

    # preprocessing operation
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # define dataloader
    data = KDD22(data_dir="dataset", split="val", val_split=0.02, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=1)

    idx = 10
    images, altitude, delta, translation = data[idx]
    img1 = images[0, :, :]
    img2 = images[1, :, :]
    print("images.shape: ", images.shape)
    print("Translation: ", translation)
    print("Altitude: ", altitude)
    print("Delta: ", delta)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow((img1 * 255).astype(int), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow((img2 * 255).astype(int), cmap="gray")
    plt.show()

