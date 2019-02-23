# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
import torch
from fc_model import NkModel
from data_load_custom_data import NkDataSet
from cnn_underba_model import Cnn_Model

def train(my_dataset_loader, model, criterion, optimizer, epoch):

    model.train()

    for i, data in enumerate(my_dataset_loader, 0):

        #Forward pass: Compute predicted y by passing x to the model

        #fc 구조 이기 때문이에 일렬로 쫙 피는 작업이 필요하다.

        images, label = data

        #그냥 imges를 하면 데이터 shape가 일치하지 않아서 에러가 난다.

     #  images = images.view(2, 30000)

        y_pred = model(images)

        #compute and print loss

        loss = criterion(y_pred, label)

        print(epoch, loss.item())

        #Zero gradients, perform a backward pass, and updata the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(my_dataset_loader, model, criterion, epoch):

    model.eval()

    for i, data in enumerate(my_dataset_loader, 0):

        #Forward pass: Compute predicited y by passing x to the model

        #fc 구조 이기 때문에 일렬로 쫙 피는 작업이 필요하다.

        images, label = data

        #그냥 images를 하면 데이터 shape가 일치하지 않아서 에러가 난다.

     #  images = images.view(2, 30000)

        print(images.size())
        print("label is label", label)

        y_pred = model(images)

        #Compute and print loss

        loss = criterion(y_pred, label)

        print(epoch, loss.item())

#Data Load

csv_path = "./file/test.csv"

custum_dataset = NkDataSet(csv_path)

print("end")

my_dataset_loader = torch.utils.data.DataLoader(dataset=custum_dataset,
                                                batch_size=2,
                                                shuffle=False,
                                                num_workers=1)


#test data set을 만들어야 한다.
#model Load
#input, hiddn, output size

D_in = 30000
H = 100
D_out = 2

model = Cnn_Model()

#CrossEnteropyLoss를 사용

criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


for epoch in range(500):

    print("epoch", epoch)

    train(my_dataset_loader, model, criterion, optimizer, epoch)
    test(my_dataset_loader, model, criterion, epoch)
