import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, groups=1):
        super().__init__()

        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s,
            padding=p, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, expansion=0.5):
        super().__init__()

        hidden = int(c2 * expansion)

        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(hidden, c2, 3)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.shortcut:
            y = y + x
        return y


class C3k2(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=False, expansion=0.5):
        super().__init__()

        hidden = int(c2 * expansion)

        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(c1, hidden, 1)

        self.blocks = nn.Sequential(
            *[
                Bottleneck(hidden, hidden, shortcut=shortcut)
                for _ in range(n)
            ]
        )

        self.cv3 = Conv(hidden * 2, c2, 1)

    def forward(self, x):
        a = self.blocks(self.cv1(x))
        b = self.cv2(x)
        return self.cv3(torch.cat([a, b], dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()

        hidden = c1 // 2

        self.cv1 = Conv(c1, hidden, 1)
        self.pool = nn.MaxPool2d(
            kernel_size=k, stride=1, padding=k // 2
        )
        self.cv2 = Conv(hidden * 4, c2, 1)

    def forward(self, x):
        x = self.cv1(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class PSA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)

        attention = torch.softmax(
            torch.bmm(q, k) / (c ** 0.5),
            dim=-1
        )

        v = self.v(x).flatten(2).transpose(1, 2)

        out = torch.bmm(attention, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        return x + self.proj(out)


class C2PSA(nn.Module):
    def __init__(self, channels, n=2):
        super().__init__()

        self.blocks = nn.Sequential(
            *[PSA(channels) for _ in range(n)]
        )

    def forward(self, x):
        return self.blocks(x)