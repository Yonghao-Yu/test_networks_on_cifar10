# test_networks_on_cifar10

## 说在前面
这个项目的目的是完成深度学习作业，我们的作业要求如下：

cifar-10分类：分别使用MLP，CNN等共三个基础网络（例如选择MLP和ResNet，VGG-Net共三个）测试学习率、优化器选择的 影响，并绘制loss曲线。<br><br>

基于这个要求所做的项目报告在[**[这里]**](实验报告.md)。

## 环境搭建
使用 anaconda 虚拟环境
```bash
conda create -n cifar10 python=3.8
```
进入环境
```bash
conda activate cifar10
```
安装 pytorch 及 timm
```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install timm
```

## train
有三个模型可选，一个是自己搭的 toy 的 mlp，第二个和第三个分别是 torchvision 官方实现的 vgg19 和 resnet34

train mlp
```bash
python main.py --batch_size 128 --epochs 100 --data_path path/to/cifar10 --model mlp --opt sgd --lr 5e-4
```
train vgg19
```bash
python main.py --batch_size 128 --epochs 100 --data_path path/to/cifar10 --model vgg19 --opt sgd --lr 5e-4
```
train resnet34
```bash
python main.py --batch_size 128 --epochs 100 --data_path path/to/cifar10 --model resnet34 --opt sgd --lr 5e-4
```
