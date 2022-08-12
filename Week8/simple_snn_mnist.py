import os
import time
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer


# 在PyTorch中搭建神经网络时，我们可以简单地使用nn.Sequential将多个网络层堆叠得到一个前馈网络，输入数据将依序流经各个网络层得到输出。
class SNN(nn.Module):
    """
    1.网络搭建：
    MNIST数据集包含若干尺寸为的8位灰度图像，总共有0~9共10个类别，因此可以先建立一个ANN网络
    ANN->SNN，只需要先去掉所有的激活函数，再将神经元添加到原来激活函数的位置，这里我们选择的是LIF神经元
    神经元之间的连接层需要用spikingjelly.activation_based.layer包装
    """

    def __init__(self, tau):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


def main():
    """
    2.指定好训练参数如学习率等以及若干其他配置
    优化器默认使用Adam，以及使用泊松编码器，在每次输入图片时进行脉冲编码
    """
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int, help='模拟时间步')
    parser.add_argument('-device', default='cuda:0', help='CPU||GPU')
    parser.add_argument('-b', default=32, type=int, help='batch大小')
    parser.add_argument('-epochs', default=20, type=int, metavar='N',
                        help='迭代次数')
    parser.add_argument('-j', default=2, type=int, metavar='N',
                        help='用于数据加载的核数')
    parser.add_argument('-data-dir', type=str, default='mnist-data', help='MNIST根目录')
    parser.add_argument('-out-dir', type=str, default='./logs', help='用于保存的根目录和检查点')
    parser.add_argument('-resume', type=str, help='从check point继续进行实验')
    parser.add_argument('-amp', action='store_true', help='自动混合精度实验')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='使用SGD或者Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='SGD的momentum')
    parser.add_argument('-lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('-tau', default=2.0, type=float, help='LIF神经元参数tau')

    # 检查结果
    args = parser.parse_args()
    print(args)
    net = SNN(tau=args.tau)
    print(net)
    # 装载至显卡
    net.to(args.device)

    """
    3.初始化数据和数据加载器
    MNIST加载的参数设置：
    ·root (string)： 表示数据集的根目录，其中根目录存在MNIST/processed/training.pt和MNIST/processed/test.pt的子目录
    ·train (bool, optional)： 如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
    ·download (bool, optional)： 如果为True，则从internet下载数据集并将其放入根目录。如果数据集已下载，则不会再次下载
    ·transform (callable, optional)： 接收PIL图片并返回转换后版本图片的转换函数
    """
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    # 自动混合精度
    # 可以在神经网络推理过程中，针对不同的层，采用不同的数据精度进行计算，
    # 从而实现节省显存和加快速度的目的
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    # 设置优化器
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']


    # 对结果进行保存
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    # 设置泊松编码器
    encoder = encoding.PoissonEncoder()

    """
    3.训练函数
    注意**：
    ①直接输出容易被噪声干扰，因此SNN的输出是输出层一段时间内的发放频率，发放率的高低表示该类别的响应大小。因此网络需要运行一段时间，即使用T个时刻后的平均发放率作为分类依据。
    ②最理想的结果是除了正确的神经元以最高频率发放，其他神经元保持静默，常采用交叉熵损失或者MSE损失，这里我们使用实际效果更好的MSE损失
    ③每次网络仿真结束后，需要重置网络状态
    """
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # 保存训练结果
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        # 保存训练结果
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        # 保存最大值
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        time_trainandtest=time.time()-start_time

        # print(args)
        # print(out_dir)
        print(
            f'epoch ={epoch},time={time_trainandtest:.4f} ,train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        # print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        # print(
        #     f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net.layer[-1]  # 输出层
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy", v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy", s_t_array)


if __name__ == '__main__':
    main()
