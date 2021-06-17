import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, channel_in, block_index):
        super().__init__()

        self.block_index = block_index

        # フィルタ数は4blockに一度倍増
        if block_index % 4 == 0:
            channel_num = channel_in * 2
            self.conv0 = nn.Conv1d(channel_in, channel_num, 1)
        else:
            channel_num = channel_in

        # 2blockに一度ダウンサンプリング
        if block_index % 2 == 1:
            down_sample = 2
        else:
            down_sample = 1

        self.maxpool = nn.MaxPool1d(down_sample, down_sample,ceil_mode=True)

        self.bn1 = nn.BatchNorm1d(channel_in)
        self.conv1 = nn.Conv1d(channel_in, channel_num, 15, padding=7, stride = down_sample) # 1/2にダウンサンプリング

        self.bn2 = nn.BatchNorm1d(channel_num)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(channel_num, channel_num, 15, padding=7)

    def forward(self, x):

        if self.block_index % 4 == 0:
            shortcut = self.maxpool(self.conv0(x))
        else:
            shortcut = self.maxpool(x)
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.dropout1(x)
        x = self.conv2(x)

        return x + shortcut

class Model_1DRes(nn.Module):
    def __init__(self, input_length = 5000):
        super(Model_1DRes, self).__init__()

        self.output_length = 20 if input_length == 5000 else 16

        self.conv1 = nn.Conv1d(12, 32, 15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)

        # block0
        self.pool = nn.MaxPool1d(1, 1, ceil_mode=True)
        self.conv0_1 = nn.Conv1d(32, 32, 15, padding=7)
        self.bn0 = nn.BatchNorm1d(32)
        self.dropout0 = nn.Dropout(p=0.2)
        self.conv0_2 = nn.Conv1d(32, 32, 15, padding=7)

        self.block1 = Block(32, 1)
        self.block2 = Block(32, 2)
        self.block3 = Block(32, 3)

        self.block4 = Block(32, 4)
        self.block5 = Block(64, 5)
        self.block6 = Block(64, 6)
        self.block7 = Block(64, 7)

        self.block8 = Block(64, 8)
        self.block9 = Block(128, 9)
        self.block10 = Block(128, 10)
        self.block11 = Block(128, 11)

        self.block12 = Block(128, 12)
        self.block13 = Block(256, 13)
        self.block14 = Block(256, 14)
        self.block15 = Block(256, 15)


        # 5000の場合: (256, 20) 4000の場合: (256, 16)
        self.bn_last = nn.BatchNorm1d(256 * self.output_length)
        self.fc1 = nn.Linear(256 * self.output_length, 1)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        #block0
        shortcut0 = self.pool(x)
        x = F.relu(self.bn0(self.conv0_1(x)))
        x = self.dropout0(x)
        x = self.conv0_2(x)
        x += shortcut0


        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = x.view(-1, 256 * self.output_length)
        x = F.relu(self.bn_last(x))
        x = self.fc1(x)
        return x