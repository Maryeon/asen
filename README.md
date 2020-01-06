# Modifying

This repository is a [PyTorch]( https://pytorch.org/ ) implementation of the paper ***Fine-Grained Fashion Similarity Learning by Attribute-Specific Embedding Network*** accepted by [AAAI 2020]( https://aaai.org/Conferences/AAAI-20/ ).

## Network

![ASEN](images/framework.png)

## Data Split

Label files in ***data*** folder give our split of train/valid/test sets and candidate/query sets of validation and test set of [FashionAI dataset]( https://tianchi.aliyun.com/competition/entrance/231671/introduction ). Raw images are not included which should be put in the ***Images*** folders according to ***label.csv***. 

## Usage

### Environments

* Pytorch 1.1.0
* CUDA 10.1.168
* Python 3.6.2

### Configure Path

You should do nothing but link the location where you place the dataset and alter the value of variable DATASET in meta.py that means which dataset the model is training on.

### Training

You can simply train the model by:

```sh
python asen.py
```

there are also optional arguments for initial learning rate, batch size and so on. Check them by 

```sh
python asen.py --help
```

### Testing

As training terminates, two snapshots are saved for testing. One is the model that has the highest performance on validation set and the other is the one of the latest epoch. You can run any of the above two models and test on the test set by:

```sh
python asen.py --test --resume ../runs/{experiment name}/xx.pth.tar
```

## Citing

If it's of any help to your research, consider citing our work:
