# Modifying

This repository is a [PyTorch]( https://pytorch.org/ ) implementation of the paper ***Fine-Grained Fashion Similarity Learning by Attribute-Specific Embedding Network*** accepted by [AAAI 2020]( https://aaai.org/Conferences/AAAI-20/ ).

## Network

![ASEN](images/framework.png)

## Data Split

Label files in ***data*** folder give our split of train/valid/test sets and candidate/query sets of validation and test set of [FashionAI dataset]( https://tianchi.aliyun.com/competition/entrance/231671/introduction ). Raw images are not included which should be put in the ***Images*** folders according to ***label.csv***. 

## Usage

### Environments

* PyTorch 1.1.0
* CUDA 10.1.168
* Python 3.6.2

We use anaconda to create our experimental environment. You can rebuild it by the following commands.

```sh
conda create -n {your_env_name} python=3.6
source activate {your_env_name}
pip install -r requirements.txt
...
conda deactivate
```

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
python asen.py --test --resume ../runs/{your_exp_name}/xx.pth.tar
```

## Citing

If it's of any help to your research, consider citing our work:
