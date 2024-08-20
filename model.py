import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, 5)
        self.conv2 = nn.Conv2d(50, 100, 3)
        self.batch_norm1 = nn.BatchNorm2d(50)
        self.batch_norm2 = nn.BatchNorm2d(100)
        self.drop = nn.Dropout2d(p = .3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(100 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.drop(x)

        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.pool(x)
        x = self.drop(x)

        # Fully connected layers
        x = t.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x