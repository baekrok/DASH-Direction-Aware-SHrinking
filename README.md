# DASH: Warm-Starting Neural Network Training Without Loss of Plasticity Under Stationarity

## Paper
paper link: https://openreview.net/pdf?id=GR5LXaglgG

## Requirements
~~~
conda environment create -f env.yaml
~~~

## Instructions

### Run
~~~
python main.py --dataset=[data]--train_type=[type]
~~~
where available data are ~~~ [data] = [cifar10, cifar100, svhn, imagenet] ~~~ and train_type are ~~~ [type]=[cold, warm, reset, l2_init, sp, dash] ~~~.

### To run Tiny-ImageNet
