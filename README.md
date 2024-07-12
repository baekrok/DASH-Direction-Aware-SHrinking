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
# To run CIFAR-10 w/ cold-starting
python main.py --dataset=cifar10 --train_type=cold
# To run CIFAR-10 w/ warm-starting
python main.py --dataset=cifar10 --train_type=warm
# To run CIFAR-10 w/ DASH
python main.py --dataset=cifar10 --train_type=dash --dash_lambda=0.3 --dash_alpha=0.3
~~~

### To run Tiny-ImageNet
