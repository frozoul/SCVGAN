# Semi-supervised complex-value GAN in Tensorflow
This is a combine of complex-value network and semi-supervised GAN, this may not have much advanced in
study research.It just a personal interest motivated practise in coding.And the code is from four part 
of already exist repositories.
There are the four part of the code:    
[SSGAN](https://github.com/gitlimlab/SSGAN-Tensorflow)  
[CV-Network_1](https://github.com/ChihebTrabelsi/deep_complex_networks)(the first CVNet code) 
[CV-Network_2](https://github.com/zsh965866221/Complex-Valued_Networks)(the second CVNet code)  
[CapsNet](https://github.com/XifengGuo/CapsNet-Keras)(Todo:this part is under develop, and may be soon to come)
## Descriptions

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 1.0.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)

## Usage

Download datasets with:
```bash
$ python download.py --dataset MNIST SVHN CIFAR10
```
Train models with downloaded datasets:
```bash
$ python trainer.py --dataset MNIST
$ python trainer.py --dataset SVHN
$ python trainer.py --dataset CIFAR10
```
Test models with saved checkpoints:
```bash
$ python evaler.py --dataset MNIST --checkpoint ckpt_dir
$ python evaler.py --dataset SVHN --checkpoint ckpt_dir
$ python evaler.py --dataset CIFAR10 --checkpoint ckpt_dir
```
The *ckpt_dir* should be like: ```train_dir/default-MNIST_lr_0.0001_update_G5_D1-20170101-194957/model-1001```

Train and test your own datasets:

* Create a directory
```bash
$ mkdir datasets/YOUR_DATASET
```

* Store your data as an h5py file datasets/YOUR_DATASET/data.hy and each data point contains
    * 'image': has shape [h, w, c], where c is the number of channels (grayscale images: 1, color images: 3)
    * 'label': represented as an one-hot vector
* Maintain a list datasets/YOUR_DATASET/id.txt listing ids of all data points
* Modify trainer.py including args, data_info, etc.
* Finally, train and test models:
```bash
$ python trainer.py --dataset YOUR_DATASET
$ python evaler.py --dataset YOUR_DATASET
```

## Acknowledgement

Part of codes is from an unpublished project with [Jongwook Choi](https://github.com/wookayin)
