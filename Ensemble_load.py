# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
import torch
from cnn_model_2 import NkModel
from M_model import LeNet
import torchvision.datasets as mdatset
import torchvision.transforms as transforms
from cnn_underba_model import Cnn_Model
from tensorboardX import SummaryWriter
import argparse
import time
import os

parser = argparse.ArgumentParser(description="PyTorch Custom Training")
parser.add_argument("--print_freq", "--p", default=2, type=int, metavar="N",
                    help="number of data loading workers (default: 4)")


parser.add_argument("--save--import torch.nn.init as initdir", dest="save_dir",
                    help="The directory used to save the trained models",
                    default="save_layer_load", type=str)


args = parser.parse_args()


def save_checkpoint(state, filename = "checkpoint.pth.bar"):

    torch.save(state, filename)


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





def test(my_dataset_loader, model, criterion, epoch, test_writer):

    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    batch_time = AverageMeter()
    end = time.time()


    for i, data in enumerate(my_dataset_loader, 0):

        images, label = data

        y_pred = model(images)

        loss = criterion(y_pred, label)

        output = y_pred.float()
        loss = loss.float()

        prec1 = accuarcy(output.data, label)[0]

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:

            print("Test : [{0}/{1}]\t"
                  "time {batch_time.val:.3f} {batch_time.avg:.3f})\t"
                  "Loss {loss.val:.3f} ({top1.avg:.3f})".format(
                i,len(my_dataset_loader), batch_time=batch_time, loss=losses, top1=top1
            ))


    print("* epoch : {epoch:.2f} prec@1 {top1.avg:.3f}"
          .format(epoch=epoch, top1=top1))

    test_writer.add_scalar("test/loss", losses.avg, epoch)
    test_writer.add_scalar("test/accuaracy", top1.avg, epoch)

    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    model_2.eval()
    model_3.eval()

    batch_time = AverageMeter()

    end = time.time()

    for i, data in enumerate(my_dataset_loader, 0):

        images, label= data

        y_pred = model(images)
        model_2_pred = model_2(images)
        model_3_pred = model_3(images)

        y_pred = (y_pred + model_2_pred + model_3_pred)/3

import torchvision.datasets as mdatset
import torchvision.transforms as transforms

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

root = "./"

train_set = mdatset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = mdatset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

model = LeNet()
model_2 = LeNet()
model_3 = LeNet()

checkpoint = torch.load("save_dir/checkpoint_0.tar")
model.load_state_dict(checkpoint["state_dict"])

checkpoint = torch.load("save_dir/checkpoint_1.tar")
model_2.load_state_dict(checkpoint["state_dict"])

checkpoint = torch.load("save_dir/checkpoint_2.tar")
model_3.load_state_dict(checkpoint["state_dict"])


criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

writer = SummaryWriter("./log")
test_writer = SummaryWriter("./log/test")

args.save_dir = "save_dir"

for epoch in range(500):

    test(test_loader, model, criterion, epoch, test_writer)

    save_checkpoint({"epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    }, filename=os.path.join(args.save_dir, "checkpoint_{}.tar".format(epoch)))
