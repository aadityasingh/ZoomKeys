
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv(in_channels, out_channels, kernel_size, stride=1, padding=1, batch_norm=False, init_zero_weights=False):
    layers = []
    conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)

def linear(in_channels, out_channels, batch_norm=False, dropout=0):
    layers = []
    layers.append(nn.Linear(in_channels, out_channels))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv1 = conv(conv_dim, conv_dim, 1,1,0)
        self.conv2 = conv(conv_dim, conv_dim, 3,1,1)
        self.conv3 = conv(conv_dim, conv_dim, 1,1,0)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv1(out)+x)
        return out

class CNN2019(nn.Module):
    def __init__(self, in_channels, num_classes, avg_pool=1, conv_dim=32, hs=[64,32], dropout=0.2):
        super(CNN2019, self).__init__()

        self.conv1 = conv(in_channels, conv_dim, 1,1,0)
        self.conv2 = conv(conv_dim, conv_dim, 3,1,1)
        self.conv3 = conv(conv_dim, conv_dim, 1,1,0)
        self.mp1 = nn.MaxPool1d(2)

        self.conv4 = conv(conv_dim, conv_dim, 1,1,0)
        self.conv5 = conv(conv_dim, conv_dim, 3,1,1)
        self.conv6 = conv(conv_dim, conv_dim, 1,1,0)
        self.mp2 = nn.MaxPool1d(2)

        self.aap = nn.AdaptiveAvgPool1d(avg_pool)

        if len(hs) < 1:
            hs = [conv_dim*avg_pool]
            self.linears = []
        elif len(hs) == 1:
            self.linears = [linear(conv_dim*avg_pool, hs[0], dropout=dropout).double()]
        else:
            self.linears = [linear(conv_dim*avg_pool, hs[0], dropout=dropout).double()]
            for i, num in enumerate(hs[1:]):
                self.linears.append(linear(hs[i], num, dropout=dropout).double())

        self.final = linear(hs[-1], num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(out)) + x # residual connection
        x = self.mp1(out)
        x = F.relu(self.conv4(x))
        out = F.relu(self.conv5(x))
        out = F.relu(self.conv6(out)) + x # residual connection
        out = self.mp2(out)
        # TODO should we add some resblock conv between or smth
        out = self.aap(out).flatten(start_dim=1)

        for layer in self.linears:
            out = F.relu(layer(out))

        # return logits for use in cross entropy loss
        return self.final(out)


class CNN2019Blast(nn.Module):
    def __init__(self, in_channels, num_classes, avg_pool=1, conv_dim=32, num_blocks=[1,1], hs=[64,32], dropout=0.2):
        super(CNN2019Blast, self).__init__()

        self.conv1 = conv(in_channels, conv_dim, 1,1,0)
        self.blocks1 = []
        for i in range(num_blocks[0]):
            self.blocks1.append(ResnetBlock(conv_dim).double())
        self.mp1 = nn.MaxPool1d(2)

        self.blocks2 = []
        for i in range(num_blocks[1]):
            self.blocks2.append(ResnetBlock(conv_dim).double())
        self.mp2 = nn.MaxPool1d(2)

        self.convl = conv(conv_dim, conv_dim, 1,1,0)

        self.aap = nn.AdaptiveAvgPool1d(avg_pool)

        if len(hs) < 1:
            hs = [conv_dim*avg_pool]
            self.linears = []
        elif len(hs) == 1:
            self.linears = [linear(conv_dim*avg_pool, hs[0], dropout=dropout).double()]
        else:
            self.linears = [linear(conv_dim*avg_pool, hs[0], dropout=dropout).double()]
            for i, num in enumerate(hs[1:]):
                self.linears.append(linear(hs[i], num, dropout=dropout).double())

        self.final = linear(hs[-1], num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        for block in self.blocks1:
            out = block(out)
        out = self.mp1(out)
        for block in self.blocks2:
            out = block(out)
        out = self.mp2(out)
        out = F.relu(self.convl(out))
        # TODO should we add some resblock conv between or smth
        out = self.aap(out).flatten(start_dim=1)

        for layer in self.linears:
            out = F.relu(layer(out))

        # return logits for use in cross entropy loss
        return self.final(out)

class CNN2019Raw(nn.Module):
    def __init__(self, in_channels, num_classes, avg_pool=1, conv_dim=32, hs=[64,32]):
        super(CNN2019Raw, self).__init__()

        self.conv1 = conv(in_channels, conv_dim, 3,1,1)
        self.conv2 = conv(conv_dim, conv_dim, 9,1,4)
        self.conv3 = conv(conv_dim, conv_dim, 3,1,1)
        self.mp1 = nn.MaxPool1d(2)

        self.conv4 = conv(conv_dim, conv_dim, 3,1,1)
        self.conv5 = conv(conv_dim, conv_dim, 9,1,4)
        self.conv6 = conv(conv_dim, conv_dim, 3,1,1)
        self.mp2 = nn.MaxPool1d(2)

        self.aap = nn.AdaptiveAvgPool1d(avg_pool)

        if len(hs) < 1:
            hs = [conv_dim*avg_pool]
            self.linears = []
        elif len(hs) == 1:
            self.linears = [linear(conv_dim*avg_pool, hs[0]).double()]
        else:
            self.linears = [linear(conv_dim*avg_pool, hs[0]).double()]
            for i, num in enumerate(hs[1:]):
                self.linears.append(linear(hs[i], num).double())

        self.final = linear(hs[-1], num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(out)) + x # residual connection
        x = self.mp1(out)
        x = F.relu(self.conv4(x))
        out = F.relu(self.conv5(x))
        out = F.relu(self.conv6(out)) + x # residual connection
        out = self.mp2(out)
        # TODO should we add some resblock conv between or smth
        out = self.aap(out).flatten(start_dim=1)

        for layer in self.linears:
            out = F.relu(layer(out))

        # return logits for use in cross entropy loss
        return self.final(out)
