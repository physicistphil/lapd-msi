import torch
import numpy as np


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        # CNN setup
        # self.seq_length = seq_length
        # self.embed_dim = out_channels = embed_dim
        # self.kernel_size = kernel_size

        self.act = torch.nn.SiLU()

        # self.conv1 = torch.nn.LazyConv2d(16, kernel_size=(5, 430), padding='valid')
        # self.conv1 = torch.nn.LazyConv2d(16, kernel_size=(5, 130), padding='valid')
        self.conv1 = torch.nn.LazyConv2d(16, kernel_size=(5, 129), padding='valid')
        self.conv2 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv3 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv4 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv5 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv6 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv7 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv8 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv9 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv10 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')

        self.dense1 = torch.nn.LazyLinear(128)
        # self.dense2 = torch.nn.LazyLinear(128)
        self.dense6 = torch.nn.LazyLinear(76)

        # Sine and cosine for positional encoding
        # self.x_sin = torch.sin(torch.arange(101) / 101 * 2 * np.pi)
        # self.x_cos = torch.cos(torch.arange(101) / 101 * 2 * np.pi)
        self.register_buffer('x_sin', torch.sin(torch.arange(101) / 101 * 2 * np.pi))
        self.register_buffer('x_cos', torch.cos(torch.arange(101) / 101 * 2 * np.pi))

    def forward(self, x):
        # x_time = x[:, 0:5353].reshape(-1, 101, 308)
        # x_notime = torch.tile(x[:, 5353:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))
        # x_time = x[:, 0:5353].reshape(-1, 101, 53)
        # x_notime = torch.tile(x[:, 5353:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))
        # x_time = x[:, 0:808].reshape(-1, 101, 8)
        # x_notime = torch.tile(x[:, 808:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))
        x_time = x[:, 0:707].reshape(-1, 101, 7)
        x_notime = torch.tile(x[:, 707:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))

        x = torch.cat([x_time, x_notime,
                       torch.tile(self.x_sin.reshape(1, 101, 1), dims=(x.shape[0], 1, 1)),
                       torch.tile(self.x_cos.reshape(1, 101, 1), dims=(x.shape[0], 1, 1))],
                      dim=2)
        # Add channels dimension
        x = torch.unsqueeze(x, dim=1)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.act(x)
        x = self.conv7(x)
        x = self.act(x)
        x = self.conv8(x)
        x = self.act(x)
        x = self.conv9(x)
        x = self.act(x)
        x = self.conv10(x)
        x = self.act(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.act(x)
        # x = self.dense2(x)
        # x = self.act(x)
        x = self.dense6(x)
        return x
