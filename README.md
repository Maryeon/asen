# Modifying

This repository is a [PyTorch]( https://pytorch.org/ ) implementation of **Attribute-Specific Embedding Network (ASEN)** proposed in our paper *Fine-Grained Fashion Similarity Learning by Attribute-Specific Embedding Network* accepted by [AAAI 2020]( https://aaai.org/Conferences/AAAI-20/ ).

## Network

![ASEN](images/framework.png)

## Requirements

### Environments

* PyTorch 1.1.0
* CUDA 10.1.168
* Python 3.6.2

We use anaconda to create our experimental environment. You can rebuild it by the following commands.

```sh
conda create -n {your_env_name} python=3.6
conda activate {your_env_name}
pip install -r requirements.txt
...
conda deactivate
```

### Download Data

#### FashionAI Dataset

As the full FashionAI has not been publicly released, we utilize its early version for the [FashionAI Global Challenge 2018](https://tianchi.aliyun.com/markets/tianchi/FashionAI). You can first sign in and download the data. Once done, you should uncompress them into the right directory:

```sh
mkdir {your_project_path}/data/fashionAI
unzip fashionAI_attributes_train1.zip fashionAI_attributes_train2.zip -d {your_project_path}/data/fashionAI
```

#### DARN Dataset

coming soon

#### DeepFashion Dataset

coming soon

#### Zappos50k Dataset

coming soon

#### Meta Data

We supply our dataset split and some descriptions of the datasets with a bunch of meta files. Download them by the following script.

```sh
wget -c -P data/fashionAI/ -i urls.txt
```

## Getting Started

All data prepared, you can simply train the model

```sh
python asen.py
```

there are also optional arguments for initial learning rate, batch size and so on. Check them by 

```sh
python asen.py --help
```

## Testing

As training terminates, two snapshots are saved for testing. One is the model that has the highest performance on validation set and the other is the one of the latest epoch. You can load any of them and test on the test set.

```sh
python asen.py --test --resume ../runs/{your_exp_name}/xx.pth.tar
```

## Citing

If it's of any help to your research, consider citing our work:
