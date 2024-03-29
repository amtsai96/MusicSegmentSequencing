from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from triplet_mnist_loader import MNIST_t # for testing
from triplet_audio_loader import TripletAudioLoader
from tripletnet import TripletNet, ChromagramEmbeddingNet, MelSpectrogramEmbeddingNet, MelSpectrogram2DEmbeddingNet
from visdom import Visdom
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Triplet Network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
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
parser.add_argument('--margin', type=float, default=0.0, metavar='M',
                    help='margin for triplet loss (default: 0.0)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')



#feature = 'mel_spec'
feature = 'mel_spec_2d'
#feature = 'chroma'
best_acc = 0
#train_split_ratio = 0.2

def main():
    global args, best_acc, plotter, device, version
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    #print('>> CUDA = {}'.format(args.cuda))
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    version = input('Enter Version:')
    args.name = args.name+'_'+version
    print(args.name)
    plotter = VisdomLinePlotter(env=args.name)

    # Load Data
    train_set = TripletAudioLoader('triplets_train.txt', feature=feature, transform=transforms.Compose([transforms.ToTensor()]))
    '''
    # Creating data indices for training and validation splits:
    train_size = int(train_split_ratio * len(train_set))
    vali_size = len(train_set) - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(train_set, [train_size, vali_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    vali_loader = DataLoader(dataset=vali_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    '''
    
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        TripletAudioLoader('triplets_test.txt',feature=feature,
        transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # Use MNIST dataset just for testing model architecture
    # train_loader = torch.utils.data.DataLoader(
    #     MNIST_t('./mnist_data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     MNIST_t('./mnist_data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)

    if feature == 'mel_spec':
        model = MelSpectrogramEmbeddingNet()
    elif feature == 'mel_spec_2d':
        model = MelSpectrogram2DEmbeddingNet()
    else: model = ChromagramEmbeddingNet()
    print('>> Model: {}\n'.format(
        model.__class__.__name__.replace('EmbeddingNet', '')))

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
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = torch.optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

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
            'epoch': epoch + 1, 'state_dict': tnet.state_dict(),
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
        dist_ap, dist_an, embedded_x, embedded_y, embedded_z = tnet(A, P, N)

        # -1 means, dist_ap should be less than dist_an
        target = torch.FloatTensor(dist_ap.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)

        loss_triplet = criterion(dist_ap, dist_an, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet #+ 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = accuracy(dist_ap, dist_an, args.margin)
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

            # # Visualization of trained flatten layer (T-SNE) (yet to be completed)
            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
            # #plot_only = 500
            # #low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            # # if args.cuda:
            # data = np.concatenate((embedded_x.cpu().detach().numpy(), embedded_y.cpu().detach().numpy()))
            # data = np.concatenate((data, embedded_z.cpu().detach().numpy()))
            # # else:
            # #     data = np.concatenate((embedded_x.detach().numpy(), embedded_y.detach().numpy()))
            # #     data = np.concatenate((data, embedded_z.detach().numpy()))
            # print(embedded_x.cpu().detach().numpy().shape) #736, 50
            # print(data.shape) #2208, 50-> fc2_out
            # #print(data)
            # X_tsne = tsne.fit_transform(data)
            # print(X_tsne.shape) #2208, 2
            # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            # X_norm = (X_tsne - x_min) / (x_max - x_min)
            # plt.figure(figsize=(8, 8))
            # for i in range(X_norm.shape[0]):
            #     plt.plot(X_norm[i, 0], X_norm[i, 1])
            #     # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
            #     #         fontdict={'weight': 'bold', 'size': 9})
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()

    print('Plot...')
    plotter.plot('acc', 'train', epoch, accs.avg, label='Accuracy')
    plotter.plot('loss', 'train', epoch, losses.avg, label='Loss')
    plotter.plot('emb_norms', 'train', epoch,
                 emb_norms.avg, label='Embeddings')

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (A, P, N) in enumerate(test_loader):
        A, P, N = A.to(device), P.to(device), N.to(device)
        A, P, N = Variable(A), Variable(P), Variable(N)

        # compute output
        dist_ap, dist_an, _, _, _ = tnet(A, P, N)
        target = torch.FloatTensor(dist_ap.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        test_loss = criterion(dist_ap, dist_an, target).data

        # measure accuracy and record loss
        acc = accuracy(dist_ap, dist_an, args.margin)
        accs.update(acc, A.size(0))
        losses.update(test_loss, A.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    print('Plot...')
    plotter.plot('acc', 'test', epoch, accs.avg, label='Accuracy')
    plotter.plot('loss', 'test', epoch, losses.avg, label='Loss')
    #plot('acc', 'test', epoch, accs.avg, color='red', label='Accuracy')
    #plot('loss', 'test', epoch, losses.avg, color='green', label='Loss')

    return accs.avg

def accuracy(dist_ap, dist_an, margin=0):
    pred = (dist_an - dist_ap - margin).cpu().data
    # print(float((pred > 0).sum()*1.0), dist_ap.size()[0])
    return float((pred > 0).sum()*1.0) / dist_ap.size()[0]

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
