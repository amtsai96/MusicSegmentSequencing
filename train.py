from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
#from triplet_mnist_loader import MNIST_t

from triplet_audio_loader import TripletAudioLoader
from simple_tripletnet import TripletNet, EmbeddingNet
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Triplet Network')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

best_acc = 0
# emb_train = []
# acc_train, acc_test = [], []
# loss_train, loss_test = [], []

def main():
    global args, best_acc, plotter, device
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    print('>> CUDA = {}'.format(args.cuda))
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)
    
    plotter = VisdomLinePlotter(env=args.name)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    base_path = './'  # './music_segments'
    train_loader = DataLoader(
        TripletAudioLoader(base_path, train=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        TripletAudioLoader(base_path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = EmbeddingNet()
    #tnet = TripletNet(model)
    tnet = TripletNet(model).to(device)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Loss and optimizer
    # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(test_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

    '''
    epochs = range(1, args.epochs+1)
    plot('acc', 'train', epochs, acc_train, color='red', label='Accuracy')
    plot('loss', 'train', epochs, loss_train, color='green', label='Loss')
    plot('emb_norms', 'train', epochs, emb_train,
         color='cyan', label='Embedding')

    plot('acc', 'test', epochs, acc_test, color='red', label='Accuracy')
    plot('loss', 'test', epochs, loss_test, color='green', label='Loss')
    print('>>> Best Accuracy: {:.4f}'.format(best_acc))
    '''


def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (A, P, N) in enumerate(train_loader):
        A, P, N = A.to(device), P.to(device), N.to(device)
        A, P, N = Variable(A), Variable(P), Variable(N)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(A, P, N)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        target = target.to(device)
        target = Variable(target)

        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        # losses.update(loss_triplet.data[0], A.size(0))
        losses.update(loss_triplet.data, A.size(0))
        accs.update(acc, A.size(0))
        # emb_norms.update(loss_embedd.data[0]/3, A.size(0))
        emb_norms.update(loss_embedd.data/3, A.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                      epoch, batch_idx * len(A), len(train_loader.dataset),
                      losses.val, losses.avg,
                      100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
    print('Plot...')
    # log avg values to somewhere
    plotter.plot('acc', 'train', epoch, accs.avg, label='Accuracy')
    plotter.plot('loss', 'train', epoch, losses.avg, label='Loss')
    plotter.plot('emb_norms', 'train', epoch,
                 emb_norms.avg, label='Embeddings')

    '''
    acc_train.append(accs.avg)
    loss_train.append(losses.avg)
    emb_train.append(emb_norms.avg)
    '''
    #return losses, accs, emb_norms

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (A, P, N) in enumerate(test_loader):
        A, P, N = A.to(device), P.to(device), N.to(device)
        A, P, N = Variable(A), Variable(P), Variable(N)

        # compute output
        dista, distb, _, _, _ = tnet(A, P, N)
        target = torch.FloatTensor(dista.size()).fill_(1)
        target = target.to(device)
        target = Variable(target)
        test_loss = criterion(dista, distb, target).data  # [0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, A.size(0))
        losses.update(test_loss, A.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    print('Plot...')
    plotter.plot('acc', 'test', epoch, accs.avg, label='Accuracy')
    plotter.plot('loss', 'test', epoch, losses.avg, label='Loss')
    #plot('acc', 'test', epoch, accs.avg, color='red', label='Accuracy')
    #plot('loss', 'test', epoch, losses.avg, color='green', label='Loss')
    '''
    acc_test.append(accs.avg)
    loss_test.append(losses.avg)
    '''

    return accs.avg


def accuracy(dista, distb, margin=0):
    pred = (dista - distb - margin).cpu().data
    # print(pred)
    #print(float((pred > 0).sum()*1.0), dista.size()[0])
    # print(pred.shape)#2760
    return float((pred > 0).sum()*1.0) / dista.size()[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' %
                        (args.name) + 'model_best.pth.tar')


'''
def plot(var_name, split_name, x, y, color, label='', version='00'):
    #('acc', 'train', epoch, accs.avg)
    # visdom ver.
    # vis.line(X=x, Y=y, win=var_name, opts={
    #          'title': '{}_{}'.format(var_name, split_name)})

    # matploylib ver.
    plt.plot(x, y, color=color, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(var_name)
    if not label == '':
        plt.legend()
    directory = "runs/%s/img/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, '{}_{}_{}.png'.format(
        version, var_name, split_name)), bbox_inches='tight', pad_inches=0.0)
    plt.close()
    plt.show()
'''


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env='main'):
        self.viz = Visdom()
        self.env = env
        self.plots = {}

    def plot(self, var_name, split_name, x, y, label):
        #('acc', 'test', epoch, accs.avg)
        if isinstance(x, torch.Tensor): x = x.cpu().numpy()
        if isinstance(y, torch.Tensor): y = y.cpu().numpy()
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=label
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array(
                [y]), env=self.env, update='append', win=self.plots[var_name], name=split_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
