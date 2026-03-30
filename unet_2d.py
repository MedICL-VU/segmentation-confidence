# local imports
import utilmods.basicutils as bu

# standard library

# third party
import torch


class Down2SkipAdd(torch.nn.Module):

    def __init__(self, nf, activation):
        super().__init__()
        activation = bu.dynamic_import(activation)

        self.front = torch.nn.Sequential(
            torch.nn.Conv2d(1, nf, 3, stride=1, padding=1),
            activation(),
            torch.nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            activation(),
        )
        self.down_1 = torch.nn.Sequential(
            torch.nn.Conv2d(nf, nf * 2, 3, stride=2, padding=1),
            activation(),
        )
        self.down_2 = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 2, nf * 4, 3, stride=2, padding=1),
            activation(),
        )
        self.bottom = torch.nn.Sequential(
            torch.nn.Conv2d(nf * 4, nf * 4, 3, stride=1, padding=1),
            activation(),
            torch.nn.Conv2d(nf * 4, nf * 4, 3, stride=1, padding=1),
            activation(),
            torch.nn.Conv2d(nf * 4, nf * 4, 3, stride=1, padding=1),
            activation(),
        )
        self.up_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                nf * 4, nf * 2, 3, stride=2, padding=1, output_padding=1),
            activation(),
        )
        self.up_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                nf * 2, nf, 3, stride=2, padding=1, output_padding=1),
            activation(),
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            activation(),
            torch.nn.Conv2d(nf, 1, 3, stride=1, padding=1),
            activation(),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        front = self.front(x)
        down_1 = self.down_1(front)
        down_2 = self.down_2(down_1)
        bottom = self.bottom(down_2)
        up_2 = self.up_2(bottom + down_2)
        up_1 = self.up_1(up_2 + down_1)
        out = self.final(up_1 + front)
        return [out]
