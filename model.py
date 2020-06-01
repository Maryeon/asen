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
    def __init__(self, backbonenet, embedding_size, n_attributes):
        super(ASENet, self).__init__()
        self.backbonenet = backbonenet
        self.n_attributes = n_attributes
        self.embedding_size = embedding_size

        self.mask_fc1 = nn.Linear(self.n_attributes, 512, bias=False)
        self.mask_fc2 = nn.Linear(self.n_attributes, 1024, bias=False)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.feature_fc = nn.Linear(1024, 1024)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, c, norm=True):
        x = self.backbonenet(x)

        attmap = self.ASA(x, c)

        x = x * attmap
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = x.sum(dim=2)

        mask = self.ACA(x, c)

        x = x * mask

        x = self.feature_fc(x)

        if norm:
            x = l2norm(x)

        return x

    def ASA(self, x, c):
        # attribute-aware spatial attention
        img_embedding = self.conv1(x)
        img_embedding = self.tanh(img_embedding)

        c = c.view(c.size(0), 1).cpu()
        mask_fc_input = torch.zeros(c.size(0), self.n_attributes).scatter_(1, c, 1)
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

        return attmap

    def ACA(self, x, c):
        # attribute-aware channel attention
        c = c.view(c.size(0), 1).cpu()
        mask_fc_input = torch.zeros(c.size(0), self.n_attributes).scatter_(1, c, 1)
        mask_fc_input = mask_fc_input.cuda()
        mask = self.relu(self.mask_fc2(mask_fc_input))
        mask = torch.cat((x, mask), dim=1)
        mask = self.fc1(mask)
        mask = self.relu(mask)
        mask = self.fc2(mask)
        mask = self.sigmoid(mask)

        return mask

    def get_heatmaps(self, x, c):
        x = self.backbonenet(x)

        attmap = self.ASA(x, c)
        attmap = attmap.squeeze()

        return attmap


class ASENet_V2(nn.Module):
    def __init__(self, backbonenet, embedding_size, n_attributes):
        super(ASENet_V2, self).__init__()
        self.backbonenet = backbonenet
        self.n_attributes = n_attributes
        self.embedding_size = embedding_size

        self.attr_embedding = torch.nn.Embedding(n_attributes, 512)

        self.attr_transform1 = nn.Linear(512, 512)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.img_bn1 = nn.BatchNorm2d(512)

        self.attr_transform2 = nn.Linear(512, 512)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1024)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, c, norm=True):
        x = self.backbonenet(x)

        attmap = self.ASA(x, c)

        x = x * attmap
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = x.sum(dim=2)

        mask = self.ACA(x, c)

        x = x * mask

        if norm:
            x = l2norm(x)

        return x

    def ASA(self, x, c):
        # attribute-aware spatial attention
        img = self.conv1(x)
        img = self.img_bn1(img)
        img = self.tanh(img)

        attr = self.attr_embedding(c)
        attr = self.attr_transform1(attr)
        attr = self.tanh(attr)
        attr = attr.view(attr.size(0), attr.size(1), 1, 1)
        attr = attr.expand(attr.size(0), attr.size(1), 14, 14)

        attmap = attr * img
        attmap = torch.sum(attmap, dim=1, keepdim=True)
        attmap = torch.div(attmap, 512 ** 0.5)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 14, 14)

        return attmap

    def ACA(self, x, c):
        # attribute-aware channel attention
        attr = self.attr_embedding(c)
        attr = self.attr_transform2(attr)
        attr = self.relu(attr)
        img_attr = torch.cat((x, attr), dim=1)
        mask = self.fc1(img_attr)
        mask = self.relu(mask)
        mask = self.fc2(mask)
        mask = self.sigmoid(mask)

        return mask

    def get_heatmaps(self, x, c):
        x = self.backbonenet(x)

        attmap = self.ASA(x, c)

        attmap = attmap.squeeze()

        return attmap

    
class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, embedding_size, n_attributes, learnedmask=True, prein=False):
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.embed_fc = nn.Linear(1024, embedding_size)
        self.avgpool = nn.AvgPool2d(14)
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_attributes, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_attributes, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_attributes)
                for i in range(n_attributes):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_attributes, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_attributes, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_attributes, embedding_size])
            mask_len = int(embedding_size / n_attributes)
            for i in range(n_attributes):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c, norm=True):
        embedded_x = self.embeddingnet(x)
        embedded_x = self.avgpool(embedded_x)
        embedded_x = embedded_x.view(embedded_x.size(0), -1)
        embedded_x = self.embed_fc(embedded_x)
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x * self.mask

        if norm:
            masked_embedding = l2norm(masked_embedding)
            
        return masked_embedding
    

model_dict = {
    'Tripletnet': Tripletnet,
    'ASENet': ASENet,
    'ASENet_V2': ASENet_V2,
    'ConditionalSimNet': ConditionalSimNet
}
def get_model(name):
    return model_dict[name]
