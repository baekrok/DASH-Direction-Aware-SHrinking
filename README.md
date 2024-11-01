# DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity

This repository contains the implementation of **Direction-Aware SHrinking (DASH)**, a method for warm-starting neural network training without losing plasticity under stationary conditions.

## 📄 Paper

For more details, check out our paper: 

[DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity](https://arxiv.org/abs/2410.23495)

## 🛠️ Setup

To set up the environment, run:

```
conda env create -f env.yaml
```

## 🚀 Usage

### Standard Training

To train the model, use:

```
python main.py --dataset [dataset] --model [model] --train_type [train_type] --optimizer_type [optimizer_type]
```

Available options:
- Datasets: `cifar10`, `cifar100`, `svhn`, `imagenet`
- Models: `resnet18`, `vgg16`, `mlp`
- Training types: `cold`, `warm`, `warm_rm`, `reset`, `l2_init`, `sp`, `dash`
- Optimizer types: `sgd`, `sam`
  
### State-of-the-Art (SoTA) Training

For SoTA settings, use:

```
python main.py --dataset [dataset] --model resnet18 --train_type [train_type] --optimizer_type [optimizer_type] \
    --sota True --weight_decay 5e-4 --learning_rate 0.1 --batch_size 128 --max_epoch 260
```

Available options for SoTA settings:
- Datasets: `cifar10`, `cifar100`, `imagenet`
- Model: `resnet18`
- Training types and optimizer types: Same as standard training


### Tiny-ImageNet Training
To use `dataset = imagenet`:

1. Download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip
or use `wget`:
~~~
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
~~~
2. Create a folder named `data`:
~~~
mkdir data
~~~
3. Unzip the downloaded Tiny-ImageNet dataset to the `data` folder 
~~~
unzip tiny-imagenet-200.zip -d data/
~~~
4. Launch `tiny-imagenet_preprocess.py` to preprocess the test data:
~~~
python tiny-imagenet_preprocess.py
~~~


## 📈 Synthetic Experiment

For our synthetic experiment described in Section 4, please refer to the `Discrete_Feature_Learning.ipynb` file.

## 📚 Citation
```bibtex
@inproceedings{shin2024dash,
    title={DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity}, 
    author={Baekrok Shin and Junsoo Oh and Hanseul Cho and Chulhee Yun},
    booktitle={Advances in Neural Information Processing Systems},
    volume={37},
    year={2024}
}
```
