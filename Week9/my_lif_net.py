import torch.nn as nn
from copy import deepcopy

from matplotlib import pyplot as plt
from spikingjelly.activation_based import layer
import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse


# 1.网络结构
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


def main():
    """
    2.指定好训练参数如学习率等以及若干其他配置
    """
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=8, type=int, help='模拟时间步')
    parser.add_argument('-device', default='cuda:0', help='CPU||GPU')
    parser.add_argument('-b', default=4, type=int, help='batch大小')
    parser.add_argument('-epochs', default=20, type=int, metavar='N',
                        help='迭代次数')
    parser.add_argument('-j', default=2, type=int, metavar='N',
                        help='用于数据加载的核数')
    parser.add_argument('-data-dir', type=str, default='D:\mydataset\DVS128Gesture', help='MNIST根目录')
    parser.add_argument('-out-dir', type=str, default='./logs_fashion_mnist', help='用于保存的根目录和检查点')
    parser.add_argument('-resume', type=str, help='从check point继续进行实验')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-amp', default=False,action='store_true', help='自动混合精度实验')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='使用SGD或者Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='SGD的momentum')
    parser.add_argument('-lr', default=0.1, type=float, help='学习率')
    parser.add_argument('-tau', default=2.0, type=float, help='LIF神经元参数tau')
    parser.add_argument('-channels', default=32, type=int, help='channels of CSNN')

    # 检查结果
    args = parser.parse_args()
    print(args)
    net = DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(),
                        detach_reset=True)
    functional.set_step_mode(net, 'm')
    print(net)
    # 装载至显卡
    net.to(args.device)

    """
    3.加载数据集
    """
    train_dataset = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                                  split_by='number')
    test_dataset = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                                 split_by='number')
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
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
    if args.cupy:
        out_dir += '_cupy'

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

    """
     3.训练函数
     注意**：
     ①直接输出容易被噪声干扰，因此SNN的输出是输出层一段时间内的发放频率，发放率的高低表示该类别的响应大小。因此网络需要运行一段时间，即使用T个时刻后的平均发放率作为分类依据。
     ②最理想的结果是除了正确的神经元以最高频率发放，其他神经元保持静默，常采用交叉熵损失或者MSE损失，这里我们使用实际效果更好的MSE损失
     ③每次网络仿真结束后，需要重置网络状态
     """
    ar_epochs = list(range(1, 21))
    train_losss = list(range(1, 21))
    test_losss = list(range(1, 21))
    ar_test_acc = list(range(1, 21))
    ar_train_acc = list(range(1, 21))
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        # 获取数据和标签
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()

            # 混合精度运算
            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            # 重置网络
            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # 保存至数组
        train_losss[epoch] = train_loss
        ar_train_acc[epoch] = train_acc

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        # 保存至数组
        test_losss[epoch] = test_loss
        ar_test_acc[epoch] = test_acc

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

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        time_trainandtest = time.time() - start_time
        # print(args)
        # print(out_dir)
        print(
            f'epoch = {epoch}, time={time_trainandtest:.4f},train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        # print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        # print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    plt.figure()
    x = ar_epochs
    y = train_losss
    # print(y)
    plt.axis([0, 21, 0, 1])
    plt.xlabel("Training steps")
    plt.ylabel("Train loss")
    plt.plot(x, y)
    plt.savefig("result_picture/trainloss2.png")
    plt.show()

    plt.figure()
    x = ar_epochs
    y = test_losss
    plt.axis([0, 21, 0, 1])
    plt.xlabel("Training steps")
    plt.ylabel("Test loss")
    plt.plot(x, y)
    plt.savefig("result_picture/testloss2.png")

    plt.figure()
    x = ar_epochs
    y = ar_test_acc
    plt.axis([0, 21, 0, 1])
    plt.xlabel("Training steps")
    plt.ylabel("Test Acc")
    plt.plot(x, y)
    plt.savefig("result_picture/testacc2.png")

    plt.figure()
    x = ar_epochs
    y = ar_train_acc
    plt.axis([0, 21, 0, 1])
    plt.xlabel("Training steps")
    plt.ylabel("Train Acc")
    plt.plot(x, y)
    plt.savefig("result_picture/trainacc2.png")


if __name__ == '__main__':
    main()
