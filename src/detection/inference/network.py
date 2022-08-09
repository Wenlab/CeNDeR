import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------- Network -----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.conv, self.shortcut = self._res_block(in_channels, out_channels, stride)

    def _res_block(self, in_channels, out_channels, stride: int = 1) -> [nn.Module, nn.Module]:
        conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),

                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
        )

        shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                nn.BatchNorm2d(out_channels),
        )
        return conv, shortcut

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x = F.relu(x1 + x2, True)
        return x


class MFDetectNetworkModule(nn.Module):

    def _head_block(self, in_channels, out_channels) -> nn.Module:
        head = nn.Sequential(
                nn.BatchNorm2d(in_channels),

                nn.Conv2d(in_channels, out_channels // 2, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(True),

                nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(True),

                nn.Conv2d(out_channels // 2, out_channels, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
        )
        return head

    @staticmethod
    def init_weights(net):
        """the weights of conv layer and fully connected layers
        are both initilized with Xavier algorithm, In particular,
        we set the parameters to random values uniformly drawn from [-a, a]
        where a = sqrt(6 * (din + dout)), for batch normalization
        layers, y=1, b=0, all bias initialized to 0.
        """
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return net


class MFDetectNetworkModule41(MFDetectNetworkModule):

    def __init__(self, num_channels):
        super().__init__()
        self.head = nn.Sequential(self._head_block(num_channels, 64), ResidualBlock(64, 64, 1), )  # 41 -> 41
        self.res1 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128, 1))  # 41 -> 21
        self.res2 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256, 1))  # 21 -> 11
        self.res3 = nn.Sequential(ResidualBlock(256, 512, 2), ResidualBlock(512, 512, 1))  # 11 -> 6
        self.res4 = nn.Sequential(ResidualBlock(512, 1024, 2), ResidualBlock(1024, 512, 1))  # 6 -> 3
        self.conv = nn.Sequential(nn.Conv2d(512, out_channels = 512, kernel_size = 3, stride = 1, padding = 0))
        self.end = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(True),
                                 nn.Dropout(0.4),
                                 )
        self.score = nn.Linear(512, 1)
        self.reg = nn.Linear(512, 4)

        MFDetectNetworkModule41.init_weights(self)

    def forward(self, x):
        x = self.head(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.conv(x)
        x = self.end(x.flatten(1))
        score, delta = torch.sigmoid(self.score(x)), self.reg(x)

        return score, delta


class MFDetectNetworkModule81(MFDetectNetworkModule):

    def __init__(self, num_channels):
        super().__init__()
        self.head = self._head_block(num_channels, 64)  # 81 -> 81
        self.res1 = ResidualBlock(64, 128, 2)  # 81 -> 41
        self.res2 = ResidualBlock(128, 256, 2)  # 41 -> 21
        self.res3 = ResidualBlock(256, 256, 2)  # 21 -> 11
        self.res4 = ResidualBlock(256, 256, 2)  # 11 -> 6
        self.res5 = ResidualBlock(256, 256, 2)  # 6 -> 3
        self.conv = nn.Sequential(
                nn.Conv2d(256, out_channels = 512, kernel_size = 3, stride = 1, padding = 0),
        )
        self.end = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 5),
        )

        MFDetectNetworkModule41.init_weights(self)

    def forward(self, x):
        x = self.head(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv(x)
        x = self.end(x.flatten(1))
        x = torch.sigmoid(x)

        return x


# ----------- Loss Function -----------
def bbox_regression_loss(pred: torch.FloatTensor, target: torch.FloatTensor
                         ) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target)

    return loss


def score_iou_loss(pred: torch.FloatTensor, target: torch.FloatTensor
                   ) -> torch.Tensor:
    """

    :param pred: It has been activated by sigmoid func
    :param target: Float target
    :return:
    """

    loss = F.binary_cross_entropy(pred, target)

    return loss


def wing_loss(pred, target, w = 0.8, epsilon = 1.0, is_half = False):
    if is_half:
        w, epsilon = torch.HalfTensor([w]).cuda(), torch.HalfTensor([epsilon]).cuda()
    else:
        w, epsilon = torch.FloatTensor([w]).cuda(), torch.FloatTensor([epsilon]).cuda()

    dis = torch.abs_(pred - target)
    isSmall = dis <= w
    isLarge = dis > w
    small_loss = w * torch.log((isSmall * dis) / epsilon + 1)
    large_loss = isLarge * dis - w * (1 - torch.log(1 + w / epsilon))
    loss = small_loss + large_loss * isLarge
    loss = torch.mean(torch.sum(loss, dim = 1), dim = 0)

    return loss
