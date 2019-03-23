# -*- coding: utf-8 -*-

import torch

'''
CODE 설명
cnn 모델을 class형으로 만듬
'''

batch_size = 2


class NkModel(torch.nn.Module):

    def __init__(self):

        super(NkModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=3, padding=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20, 40, kernel_size=3, padding=1, stride=1)

        # 이 부분을 고쳐 줘야 한다.

        self.conv3 = torch.nn.Conv2d(40, 60, kernel_size=3, padding=1, stride=1)
        self.conv4 = torch.nn.Conv2d(60, 100, kernel_size=15, padding=1, stride=1)

        self.linear1 = torch.nn.Linear(25600,100)
        self.linear2 = torch.nn.Linear(100,10)

    def forward(self, x):
       # print("first x", x.size())

        x = self.conv1(x)

       # print("conv1 ", x.size())

        x = self.relu(x)
        x = self.conv2(x)

       # print("conv2", x.size())

        x = self.relu(x)
        x = self.conv3(x)

      #  print("conv3", x.size())

        x = self.conv4(x)

       # print("conv4", x.size())

        x = x.view(100,-1)

        #print(x.size())

        # print("conv3",x.size())

        # print("last x size",x.size())

        x = self.linear1(x)
        x = self.linear2(x)

        return x
