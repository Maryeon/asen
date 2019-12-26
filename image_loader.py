import os
import csv
import random
import pickle
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from meta import *
from PIL import Image
from torch.autograd import Variable


category = CONDITIONS
category_num = CATEGORY_NUM


def make_dataset(num_triplets):

    filename_csv = os.path.join(DATAPATH, DATASET, LABEL_FILE['train'])
    csvFile = open(filename_csv,'r')
    reader = csv.reader(csvFile)
    data = []
    for item in reader:
        data.append(item)
    csvFile.close()


    category_dict = {}

    triplets = []
    for c in category:
        category_dict[c] = []



    for item in data:
        if item[2].find('m') == -1:
            category_dict[item[1].replace('_labels','')].append([item[0], item[2].find('y')])

    #print('data generation')
    for i in range(num_triplets):
        cate_r = random.randint(0, len(category)-1)

        cate_sub = random.randint(0, category_num[category[cate_r]]-1)

        while True:
            a = random.randint(0, len(category_dict[category[cate_r]])-1)
            if category_dict[category[cate_r]][a][1] == cate_sub:
                break

        while True:
            b = random.randint(0, len(category_dict[category[cate_r]])-1)
            if  category_dict[category[cate_r]][b][1] != cate_sub:
                break
            
        while True:
            c = random.randint(0, len(category_dict[category[cate_r]])-1)
            if a != c and category_dict[category[cate_r]][c][1] == cate_sub:
                break
            
        triplets.append([category_dict[category[cate_r]][a],category_dict[category[cate_r]][b],category_dict[category[cate_r]][c],cate_r])

    return triplets


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, base_path, num_triplets, transform=None,
                 loader=default_image_loader):
        self.num_triplets = num_triplets
        self.triplets = make_dataset(self.num_triplets)
        self.dataroot = os.path.join(root, base_path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        path1 = self.triplets[index][0][0]
        path2 = self.triplets[index][1][0]
        path3 = self.triplets[index][2][0]
        c = self.triplets[index][3]

        if os.path.exists(os.path.join(self.dataroot, path1)):
            img1 = self.loader(os.path.join(self.dataroot, path1))
        else:
            return None

        if os.path.exists(os.path.join(self.dataroot, path2)):
            img2 = self.loader(os.path.join(self.dataroot, path2))
        else:
            return None

        if os.path.exists(os.path.join(self.dataroot, path3)):
            img3 = self.loader(os.path.join(self.dataroot, path3))
        else:
            return None

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3, c

    def __len__(self):
        return len(self.triplets)


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, base_path, filenames_filename, split, cand_query, transform=None, loader=default_image_loader):
        ''' root: rootpath to data
            base_path: dataset, e.g. fashionAI
            filenames_filename: file of image names
            split:  valid or test
            cand_query: candidate or query
        '''
        self.root = root
        self.base_path = base_path
        self.filenamelist = []
        with open(os.path.join(self.root, filenames_filename)) as f:
            for line in f:
                self.filenamelist.append(line.rstrip('\n'))
        samples = []
        with open(os.path.join(self.root, cand_query+'_'+split+'.txt')) as f:
            for line in f:
                samples.append((line.split()[0], int(line.split()[1]), int(line.split()[2])))   #picid condition groundtruth
        np.random.shuffle(samples)
        self.samples = samples
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, c, gdtruth = self.samples[index]
        if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path)])):
            img = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path)]))
            if self.transform is not None:
                img = self.transform(img)
            return img, c, gdtruth, self.filenamelist[int(path)].strip().split('/')[-1].split('.')[0]   #imgdata, condition, groundtruth, imgid(for retrieval use)
        else:
            return None

    def __len__(self):
        return len(self.samples)

    def sample(self):
        samples = []
        for condition in range(len(category)):
            sub = [sample for sample in self.samples if sample[1] == condition]
            sample = random.sample(sub, 2)
            samples.append((os.path.join(self.root, self.base_path, self.filenamelist[int(sample[0][0])]), sample[0][1]))
            samples.append((os.path.join(self.root, self.base_path, self.filenamelist[int(sample[1][0])]), sample[1][1]))

        return samples



def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    conditions = [1,2,3,4,5,6,7]
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_query_loader = torch.utils.data.DataLoader(
        ImageLoader('../data', 'fashionAI', 'filenames_test.txt', 
            'test', 'query',
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=32, shuffle=True, **kwargs)
    samples = test_query_loader.dataset.sample()
    #print(cand_set.size())


if __name__ == '__main__':
    main()