import torch.nn as nn
import torch.nn.functional as F

class Model_2D(nn.Module):
    def __init__(self, input_length = 5000):
        super(Model_2D, self).__init__()

        self.output_length = 8 if input_length == 5000 else 7

        self.conv1 = nn.Conv2d(1, 16, (1, 5))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((1, 2), stride=(1,2))

        self.conv2 = nn.Conv2d(16, 16, (1, 5))
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((1, 2), stride=(1,2))

        self.conv3 = nn.Conv2d(16, 32, (1, 5))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d((1, 4), stride=(1,4))

        self.conv4 = nn.Conv2d(32, 32, (1, 3))
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d((1, 2), stride=(1,4))

        self.conv5 = nn.Conv2d(32, 64, (1, 3))
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d((1, 2), stride=(1,2))

        self.conv6 = nn.Conv2d(64, 64, (1, 3))
        self.bn6 = nn.BatchNorm2d(64)
        self.pool6 = nn.MaxPool2d((1, 4), stride=(1,4))

        self.conv7 = nn.Conv2d(64, 64, (12, 1))
        self.bn7 = nn.BatchNorm2d(64)


        self.fc1 = nn.Linear(64 * 1 * self.output_length, 256) # 5000の場合 : (64, 1, 8), 4000の場合 : (64, 1, 7)
        self.bn8 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 1)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        x = F.relu(self.bn7(self.conv7(x)))

        x = x.view(-1, 64 * 1 * self.output_length)
        x =self.drop1( F.relu(self.bn8(self.fc1(x))))
        x =self.drop2( F.relu(self.bn9(self.fc2(x))))
        x =self.fc3(x)
        return x