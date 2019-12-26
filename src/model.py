from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which attribute images are compared"""
        embedded_x = self.embeddingnet(x, c)
        embedded_y = self.embeddingnet(y, c)
        embedded_z = self.embeddingnet(z, c)
        sim_a = torch.sum(embedded_x * embedded_y, dim=1)
        sim_b = torch.sum(embedded_x * embedded_z, dim=1)

        return sim_a, sim_b


class ASENet(nn.Module):
    def __init__(self, backbonenet, embedding_size, n_conditions):
        super(ASENet, self).__init__()
        self.backbonenet = backbonenet
        self.n_conditions = n_conditions
        self.embedding_size = embedding_size

        self.mask_fc1 = nn.Linear(self.n_conditions, 512, bias=False)
        self.mask_fc2 = nn.Linear(self.n_conditions, 1024, bias=False)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.feature_fc = nn.Linear(1024, 1024)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, task, norm=True):
        x, _ = self.backbonenet(x)

        img_embedding = self.conv1(x)
        img_embedding = self.tanh(img_embedding)

        c = task.view(task.size(0), 1).cpu()
        mask_fc_input = torch.zeros(c.size(0), self.n_conditions).scatter_(1, c, 1)
        mask_fc_input = mask_fc_input.cuda()
        mask = self.mask_fc1(mask_fc_input)
        mask = self.tanh(mask)
        mask = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask = mask.expand(mask.size(0), mask.size(1), 14, 14)

        #spatial attention
        attmap = mask * img_embedding
        attmap = self.conv2(attmap)
        attmap = self.tanh(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 14, 14)

        x = x * attmap
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = x.sum(dim=2)

        #channel attention
        mask = self.relu(self.mask_fc2(mask_fc_input))
        mask = torch.cat((x, mask), dim=1)
        mask = self.fc1(mask)
        mask = self.relu(mask)
        mask = self.fc2(mask)
        mask = self.sigmoid(mask)
        x = x * mask
        x = self.feature_fc(x)

        if norm:
            x = l2norm(x)

        return x

    def get_heatmaps(self, x, task):
        feature, _ = self.backbonenet(x)

        img_embedding = self.conv1(feature)
        img_embedding = self.tanh(img_embedding)

        task = task.view(task.size(0), 1).cpu()
        mask_fc_input = torch.zeros(task.size(0), self.n_conditions).scatter_(1, task, 1)
        mask_fc_input = mask_fc_input.cuda()
        mask = self.mask_fc1(mask_fc_input)
        mask = self.tanh(mask)
        mask = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask = mask.expand(mask.size(0), mask.size(1), 14, 14)

        attmap = mask * img_embedding
        attmap = self.conv2(attmap)
        attmap = self.tanh(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 14, 14)
        attmap = attmap.squeeze()
        return attmap