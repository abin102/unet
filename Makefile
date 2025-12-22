PY=python

train-mnist-resnet:
	$(PY) train.py --cfg configs/mnist_resnet18.yaml

train-mnist-densenet:
	$(PY) train.py --cfg configs/mnist_densenet121.yaml

train-cifar-resnet:
	$(PY) train.py --cfg configs/cifar10_resnet18.yaml

train-cifar-densenet:
	$(PY) train.py --cfg configs/cifar10_densenet121.yaml
