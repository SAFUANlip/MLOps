import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, inputs):
        x = self.block(inputs)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chan=3, n_clas=3):
        super().__init__()
        """ Encoder """
        self.e0 = nn.Conv2d(in_channels=in_chan, out_channels=3, kernel_size=1)
        self.e1 = encoder_block(3, 8)
        self.e2 = encoder_block(8, 16)
        self.e3 = encoder_block(16, 32)
        self.e4 = encoder_block(32, 64)
        # self.e5 = encoder_block(64, 128)

        """ Bottleneck """
        self.b = conv_block(64, 128)
        self.b2 = conv_block(128, 256)
        self.b3 = conv_block(256, 128)
        # self.b4 = conv_block(256, 128)

        """ Decoder """
        # self.d0 = decoder_block(256, 128)
        self.d1 = decoder_block(128, 64)
        self.d2 = decoder_block(64, 32)
        self.d3 = decoder_block(32, 16)
        self.d4 = decoder_block(16, 8)

        """ Classifier """
        self.outputs = nn.Conv2d(8, n_clas, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s0 = self.e0(inputs)
        s1, p1 = self.e1(s0)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # s5, p5 = self.e5(p4)

        """ Bottleneck """
        b = self.b(p4)
        b = self.b2(b)
        b = self.b3(b)
        # b = self.b4(b)

        """ Decoder """
        # d0 = self.d0(b, s5)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        return outputs
