# DASH: Warm-Starting Neural Network Training Without Loss of Plasticity Under Stationarity

This repository contains the implementation of Direction-Aware SHrinking (DASH), a method for warm-starting neural network training without losing plasticity under stationary conditions.

## 📄 Paper

For more details, check out our paper: 
[DASH: Warm-Starting Neural Network Training Without Loss of Plasticity Under Stationarity](https://openreview.net/pdf?id=GR5LXaglgG)

## 🛠️ Setup

To set up the environment, run:

```
conda env create -f env.yaml
```

## 🚀 Usage

### Basic Training

To train the model, use:

```
python main.py --dataset [dataset] --train_type [type]
```

Available options:
- Datasets: `cifar10`, `cifar100`, `svhn`, `imagenet`
- Training types: `cold`, `warm`, `reset`, `l2_init`, `sp`, `dash`

### Training on Tiny-ImageNet
