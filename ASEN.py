from __future__ import print_function,division
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from image_loader import TripletImageLoader, ImageLoader
from visdom import Visdom
import cv2
from PIL import Image
import matplotlib.cm as cm
import numpy as np
import resnet
from model import ASENet
from config import *
from metric import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=100000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--dim_embed', type=int, default=1024, metavar='N',
                    help='how many dimensions in embedding (default: 1024)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--conditions', nargs='*', type=int,
                    help='Set of similarity notions')
parser.add_argument('--visdom_port', type=int, default=8098, metavar='N',
                    help='visdom port')
parser.set_defaults(test=False)
parser.set_defaults(learned=False)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_mAP = 0


def main():
    global args, best_mAP
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.visdom:
        global plotter 
        plotter = VisdomLinePlotter(env_name=args.name)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    global conditions
    if args.conditions is not None:
        conditions = args.conditions
    else:
        conditions = [0,1,2,3,4,5,6,7]
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    test_candidate_loader = torch.utils.data.DataLoader(
        ImageLoader('../data', 'fashionAI', 'filenames_test.txt', 
            'test', 'candidate',
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_query_loader = torch.utils.data.DataLoader(
        ImageLoader('../data', 'fashionAI', 'filenames_test.txt', 
            'test', 'query',
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_candidate_loader = torch.utils.data.DataLoader(
        ImageLoader('../data', 'fashionAI', 'filenames_valid.txt', 
            'valid', 'candidate',
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_query_loader = torch.utils.data.DataLoader(
        ImageLoader('../data', 'fashionAI', 'filenames_valid.txt', 
            'valid', 'query',
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    model = resnet.resnet50_feature()
    global smn_model
    smn_model = get_model('ASENet')(model, n_conditions=len(conditions), embedding_size=args.dim_embed)

    tnet = get_model('Tripletnet')(smn_model)
    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {} mAP {})"
                    .format(args.resume, checkpoint['epoch']-1, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    parameters = filter(lambda p: p.requires_grad, tnet.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        test_mAP = test(test_candidate_loader, test_query_loader, smn_model)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train_loader = reload(normalize, kwargs)
        train(train_loader, tnet, criterion, optimizer, epoch)
        # evaluate on validation set
        mAP = test(val_candidate_loader, val_query_loader, smn_model, epoch)

        # remember best acc and save checkpoint
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_mAP,
        }, is_best)


def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
        data1, data2, data3, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)

    # compute output
        sim_a, sim_b = tnet(data1, data2, data3, c)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(sim_a.size()).fill_(-1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(sim_a, sim_b, target)
        loss = loss_triplet

        # measure accuracy and record loss
        acc = accuracy(sim_a, sim_b)
        losses.update(loss.data.item(), data1.size(0))
        accs.update(acc, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%)'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg))

    # log avg values to visdom
    if args.visdom:
        plotter.plot('acc', 'train', epoch, accs.avg)
        plotter.plot('loss', 'loss', epoch, losses.avg)


def test(test_candidate_loader, test_query_loader, test_model, epoch=-1):
    mAPs = AverageMeter()
    mAP_cs = {}
    for condition in conditions:
        mAP_cs[condition] = AverageMeter()

    # switch to evaluation mode
    test_model.eval()

    cand_set = [[] for _ in conditions]
    c_gdtruth = [[] for _ in conditions]
    for _, (img, c, gdtruth, _) in enumerate(test_candidate_loader):
        
        if args.cuda:
            img, c, gdtruth = img.cuda(), c.cuda(), gdtruth.cuda()
        img, c, gdtruth = Variable(img), Variable(c), Variable(gdtruth)

        # compute output
        masked_embedding = test_model(img, c)
        for i in range(masked_embedding.size(0)):
            cand_set[c[i].data.item()].append(masked_embedding[i].cpu().data.numpy())
            c_gdtruth[c[i].data.item()].append(gdtruth[i].cpu().data.item())
    for condition in conditions:
        cand_set[condition] = np.array(cand_set[condition])
        c_gdtruth[condition] = np.array(c_gdtruth[condition])
        #print(cand_set[condition].shape)
        #print(c_gdtruth[condition].shape)

    queries = [[] for _ in conditions]
    q_gdtruth = [[] for _ in conditions]
    for _, (img, c, gdtruth, _) in enumerate(test_query_loader):
        if args.cuda:
            img, c, gdtruth = img.cuda(), c.cuda(), gdtruth.cuda()
        img, c, gdtruth = Variable(img), Variable(c), Variable(gdtruth)

        masked_embedding = test_model(img, c)
        for i in range(masked_embedding.size(0)):
            queries[c[i].data.item()].append(masked_embedding[i].cpu().data.numpy())
            q_gdtruth[c[i].data.item()].append(gdtruth[i].cpu().data.item())
    for condition in conditions:
        queries[condition] = np.array(queries[condition])
        q_gdtruth[condition] = np.array(q_gdtruth[condition])
    
    for condition in conditions:
        mAP = mean_average_precision(cand_set[condition], queries[condition], c_gdtruth[condition], q_gdtruth[condition])
        mAPs.update(mAP, queries[condition].shape[0])
        mAP_cs[condition].update(mAP)


    print('Train Epoch: {}'.format(epoch))
    for condition in conditions:
        print('{} mAP: {:.4f}'.format(CONDITIONS[condition], 100. * mAP_cs[condition].val))
    print('MeanAP: {:.4f}\n'.format(100. * mAPs.avg))

    if args.visdom:
        samples = test_candidate_loader.dataset.sample()
        imgs = []
        x = []
        for sample in samples:
            img = cv2.imread(sample[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            origin_img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
            imgs.append(origin_img)

            input_img = Image.fromarray(img)
            input_img = test_candidate_loader.dataset.transform(input_img)
            x.append(input_img.numpy())

        tasks = [sample[1] for sample in samples]

        tasks = np.array(tasks)
        x = np.array(x)
        c = torch.from_numpy(tasks)
        x = torch.from_numpy(x)
        x, c = x.cuda(), c.cuda()
        heatmaps = test_model.get_heatmaps(x, c)
        plotter.plot_attention(imgs, heatmaps.cpu().data.numpy(), tasks)
        plotter.plot('mAP', 'valid', epoch, mAPs.avg)

    return mAPs.avg


def reload(normalize,kwargs):
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('../data', 'fashionAI/train',
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "../runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../runs/%s/'%(args.name) + 'model_best.pth.tar')


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=args.visdom_port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name, update='append')

    def plot_attention(self, imgs, heatmaps, tasks, alpha=0.5):
        for i in range(len(tasks)):
            heatmap = heatmaps[i]
            heatmap = cv2.resize(heatmap, (224,224), interpolation=cv2.INTER_CUBIC)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmap_marked = np.uint8(cm.gist_rainbow(heatmap)[..., :3] * 255)
            heatmap_marked = cv2.cvtColor(heatmap_marked, cv2.COLOR_BGR2RGB)
            heatmap_marked = np.uint8(imgs[i] * alpha + heatmap_marked * (1. - alpha))
            heatmap_marked = heatmap_marked.transpose([2,0,1])

            win_name = 'img %d - %s'%(i,CONDITIONS[tasks[i]])
            if win_name not in self.plots:
                self.plots[win_name] = self.viz.image(
                    heatmap_marked,
                    env=self.env,
                    opts=dict(
                        title=win_name
                    )
                )
                self.plots[win_name+'heatmap'] = self.viz.heatmap(
                    heatmap,
                    env=self.env,
                    opts=dict(
                        title=win_name
                    )
                )
            else:
                self.viz.image(
                    heatmap_marked,
                    env=self.env,
                    win =self.plots[win_name],
                    opts=dict(
                        title=win_name
                    )
                )
                self.viz.heatmap(
                    heatmap,
                    env=self.env,
                    win=self.plots[win_name+'heatmap'],
                    opts=dict(
                        title=win_name
                    )
                )


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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(sim_a, sim_b):
    margin = 0
    pred = (sim_b - sim_a - margin).cpu().data
    return float((pred > 0).sum())/float(sim_a.size()[0])


def mean_average_precision(cand_set, queries, c_gdtruth, q_gdtruth):
    '''
    calculate mAP of a conditional set. Samples in candidate and query set are of the same condition.
        cand_set: 
            type:   nparray
            shape:  c x feature dimension
        queries:
            type:   nparray
            shape:  q x feature dimension
        c_gdtruth:
            type:   nparray
            shape:  c
        q_gdtruth:
            type:   nparray
            shape:  q
    '''
 
    scorer = APScorer(cand_set.shape[0])

    simmat = np.matmul(queries, cand_set.T)
    #similarity matrix

    ap_sum = 0
    for q in range(simmat.shape[0]):
        sim = simmat[q]
        index = np.argsort(-sim)
        sorted_labels = []
        for i in range(index.shape[0]):
            if c_gdtruth[index[i]] == q_gdtruth[q]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)
        
        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP = ap_sum / simmat.shape[0]

    return mAP


if __name__ == '__main__':
    main()
