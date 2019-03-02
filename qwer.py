# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
import torch
from cnn_underba_model import Cnn_Model
from my_dataset import NkDataSet
from tensorboardX import SummaryWriter

def accuarcy(output, target, topk = (1,)):

    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:

        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))


    return res


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):

        self.val = val
        self.sum += val * n #sum = sum + val * n
        self.count += n
        self.avg = self.sum / self.count


def train(my_dataset_loader, model, criterion, optimizer, epoch, writer):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    for i, data in enumerate(my_dataset_loader, 0):

        #Forward pass: computer predicited y by passing x to the model

        #fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.

        images, label = data

        #images = torc.autograd.Variable(images)
        #label = torch.autograd.Variable(label)

        #그냥 images를 하면 데이터 shape가 일치하지 않아서 에러가 난다.

        y_pred = model(images)

        #Compute and print loss

        loss = criterion(y_pred, label)

        #print(epoch, loss, item()

        #Zero gradients, perform a backward pass, and update the weights

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = y_pred.float()
        loss = loss.float()

        prec1 = accuarcy(output.data, label)[0]

        prec2 = accuarcy(output.data, label)

        #print("prec1", (prec1))
        #print("prec2", (prec2))

        #print("loss.item", loss)
        #print("real loss item", loss.item()

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    writer.add_scalar("Train/loss", losses.avg, epoch)
    writer.add_scalar("Train/accuaracy", top1.avg, epoch)


def test(my_dataset_loader, model, criterion, epoch, test_writer):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    for i, data in enumerate(my_dataset_loader, 0):

        images, label = data

        y_pred = model(images)

        loss = criterion(y_pred, label)

        output = y_pred.float()
        loss = loss.float()

        prec1 = accuarcy(output.data, label)[0]

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    print("* epoch : {epoch:.2f} prec@1 {top1.avg:.3f}"
          .format(epoch=epoch, top1=top1))


csv_path = "./file/data_load.csv"

custom_dataset = NkDataSet(csv_path)


my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=5,
                                                shuffle=False,
                                                num_workers=1)

D_in = 30000
H = 100
D_out = 2

model = Cnn_Model()

criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

writer = SummaryWriter("./log")
test_writer = SummaryWriter("./log/test")


for epoch in range(500):

    train(my_dataset_loader, model, criterion, optimizer, epoch, writer)
    test(my_dataset_loader, model, criterion, epoch, test_writer)

