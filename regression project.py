# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
import torch
from regression_model import Regression_model
from my_dataset import NkDataSet
from tensorboardX import SummaryWriter
import argparse
import set_variable
import time
import os
import torchvision.datasets as mdatset
import torchvision.transforms as transforms
from regression_model import Regression_model

#argparse 커맨드로 부터 인자를 받아서 수행 할 수 있는 기능. fault로 기본값을 지정할 수 도 있다.

parser = argparse.ArgumentParser(description='PyTorch Custom Training')
parser.add_argument('--print_freq', '--p', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4')
parser.add_argument('--save-import torch.nn.init as initdir',dest='save_dir',
                    help='The directory used to save the trained models',default='save_layer_load',type=str)
#metavar 는 nickname 같은 역할이라고 생각을 하면 된다.

#dest 입력되는 값이 저장되는 변수

args = parser.parse_args()

#save checkpoint

def adjust_learning_rate(optimizer, epoch, lr):

    lr = lr * (0.1 ** (epoch // 50))

    for param_group in optimizer.param_groups:

        param_group["lr"] = lr

def save_checkpoint(state,filename='checkpoint.pth.bar'):
    torch.save(state,filename)

def accuracy(output, target, topk=(1,)):
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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(my_dataset_loader,model,criterion,optimizer,epoch,writer):
    model.train()
    losses = AverageMeter()
    for i, data in enumerate(my_dataset_loader, 0):
        # Forward pass: Compute predicted y by passing x to the model
        # fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.
        images, label = data
        images = torch.autograd.Variable(images)
        label = torch.autograd.Variable(label).float()
        # 그냥 images 를 하면 에러가 난다. 데이터 shape 이 일치하지 않아서
        y_pred = model(images)
        # Compute and print loss
        loss = criterion(y_pred, label)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.float()
        losses.update(loss.item(), images.size(0))

    writer.add_scalar('Train/loss', losses.avg, epoch)

def test(my_dataset_loader, model, criterion, epoch, test_writer):
    losses = AverageMeter()
    model.eval()
    for i, data in enumerate(my_dataset_loader, 0):
        # Forward pass: Compute predicted y by passing x to the model
        # fc 구조 이기 때문에 일렬로 쫙피는 작업이 필요하다.
        images, label = data
        # 그냥 images 를 하면 에러가 난다. 데이터 shape 이 일치하지 않아서
        y_pred = model(images)
        label = label.float()
        # Compute and print loss
        loss = criterion(y_pred, label)
        loss = loss.float()
        losses.update(loss.item(), images.size(0))

        if i % args.print_freq == 0:
            print('Test : [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i,len(my_dataset_loader),loss=losses
            ))
    print('*, epoch : {epoch:.2f} Prec@1 {losses.avg:.3f}'.format(epoch=epoch,losses=losses))


    test_writer.add_scalar('Test/loss', losses.avg, epoch)

    return losses.avg

#Data_Load
csv_path = './file/train.csv'
custom_dataset = NkDataSet(csv_path)
my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=set_variable.batch_size,
                                                shuffle=True, num_workers=1)


csv_path = './file/test.csv'
custom_dataset = NkDataSet(csv_path)
my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=set_variable.batch_size,
                                                shuffle=True, num_workers=1)

model = Regression_model()
print(model)
#Regression 이기 때문에 loss가 변경 되어야 한다.
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
writer = SummaryWriter('./log')
test_writer = SummaryWriter('./log/test')

best_prec = 0

save_dir = './save_dir'
lr = 1e-3

for epoch in range(500):

    adjust_learning_rate(optimizer,epoch,lr)

    train(my_dataset_loader, model, criterion, optimizer, epoch, writer)

    if(epoch == 0):

        prec = test(my_dataset_loader, model, criterion, epoch, test_writer)
        best_prec = prec

    else:
        prec = test(my_dataset_loader,model,criterion,epoch,test_writer)

    if(prec < best_prec):

        best_epoch = epoch
        best_prec = prec
        save_checkpoint({
            "epoch":epoch + 1,
            "state_dict":model.state_dict(),
            "best_prec1":best_prec,
            "best_epoch":best_epoch
        }, filename=os.path.join(save_dir,"checkpoint_{}.tar".format(epoch)))



