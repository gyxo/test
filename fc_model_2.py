# -*- coding: utf-8 -*-

import torch

'''
CODE 설명
fc 모델을 class형으로 만듬
'''

class FcModel(torch.nn.Module):

    def __init__(self):

        super(FcModel, self).__init__()

        self.linear1 = torch.nn.Linear(784,20)
        self.linear2 = torch.nn.Linear(20,40)
        self.linear3 = torch.nn.Linear(40,60)
        self.linear4 = torch.nn.Linear(60,80)
        self.linear5 = torch.nn.Linear(80,10)

    def forward(self, x):

        x = x.view(100,-1)

        print(x.size())

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        return x
