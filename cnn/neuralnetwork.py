import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(25, 75, 3, padding=1)
        self.conv2 = nn.Conv2d(75, 150, 3, padding=1)
        self.conv3 = nn.Conv2d(150, 300, 3, padding=1)
        self.conv4 = nn.Conv2d(300, 600, 3, padding=1)

        self.fc1 = nn.Linear(38400, 1024)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1024, 4096)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 34800)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)

        return x
