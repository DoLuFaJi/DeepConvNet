import torch.nn as nn
import torch.nn.functional as F

DEBUG_FORWARD = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 14, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7*7*14, 84)
        # self.fc2 = nn.Linear(120, 84)
        # 2 outputs pas 10
        # self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(84, 2)

    def forward(self, x):
        if DEBUG_FORWARD:
            print(x.size())
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        if DEBUG_FORWARD:
            print(x.size())

        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        if DEBUG_FORWARD:
            print(x.size())

        x = x.view(-1, self.num_flat_features(x))

        if DEBUG_FORWARD:
            print(x.size())

        x = F.relu(self.fc1(x))

        if DEBUG_FORWARD:
            print(x.size())

        # x = F.relu(self.fc2(x))
        #
        # if DEBUG_FORWARD:
        #     print(x.size())
        #
        # x = F.relu(self.fc3(x))
        #
        # if DEBUG_FORWARD:
        #     print(x.size())

        x = self.fc4(x)
        if DEBUG_FORWARD:
            print(x.size())
        if DEBUG_FORWARD:
            import pdb; pdb.set_trace()
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetTuto(nn.Module):
    def __init__(self):
        super(NetTuto, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(576, 84)
        # self.fc2 = nn.Linear(120, 84)
        # 2 outputs pas 10
        # self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(84, 2)

    def forward(self, x):
        if DEBUG_FORWARD:
            print(x.size())

        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        if DEBUG_FORWARD:
            print(x.size())

        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))

        if DEBUG_FORWARD:
            print(x.size())

        x = x.view(-1, self.num_flat_features(x))

        if DEBUG_FORWARD:
            print(x.size())

        x = F.relu(self.fc1(x))

        if DEBUG_FORWARD:
            print(x.size())

        # x = F.relu(self.fc2(x))
        #
        # if DEBUG_FORWARD:
        #     print(x.size())
        #
        # x = F.relu(self.fc3(x))
        #
        # if DEBUG_FORWARD:
        #     print(x.size())

        x = self.fc4(x)
        if DEBUG_FORWARD:
            print(x.size())
        if DEBUG_FORWARD:
            import pdb; pdb.set_trace()
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
