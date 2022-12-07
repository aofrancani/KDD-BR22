import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class CNNVO(nn.Module):
    def __init__(self):
        super(CNNVO, self).__init__()
        self.doubleConv1 = DoubleConv(2, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.doubleConv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.doubleConv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.doubleConv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.doubleConv5 = DoubleConv(256, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.doubleConv6 = DoubleConv(512, 512)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fc1 = nn.Linear(in_features=512*4*4 + 2, out_features=2)

    def forward(self, x, h, delta):
        x = self.doubleConv1(x)
        x = self.pool1(x)
        x = self.doubleConv2(x)
        x = self.pool2(x)
        x = self.doubleConv3(x)
        x = self.pool3(x)
        x = self.doubleConv4(x)
        x = self.pool4(x)
        x = self.doubleConv5(x)
        x = self.pool5(x)
        x = self.doubleConv6(x)
        x = self.flatten(x)
        x = torch.cat((x, h, delta), dim=1)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    # build and load model
    model = CNNVO()
    print(model)