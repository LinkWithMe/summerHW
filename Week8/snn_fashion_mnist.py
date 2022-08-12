import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing


# Fashion-MNIST与MNIST数据集的格式相同，均为1*28*28灰度图片
# 1.网络结构定义
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        """

        :param T: 模拟时间步T
        :param channels:通道数
        :param use_cupy:是否使用cupy
        """
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
            # 卷积层
            # 网络输入通道数1
            # 网络输出通道数channels
            # 卷积核大小kernel_size
            # 填充padding
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            # 批量标准化操作
            # 参数为输入图像的通道数channels
            layer.BatchNorm2d(channels),
            # 替换函数设置为surrogate.ATan()
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            # 最大值池化
            # 最大池化大小为2
            # 步长为2
            layer.MaxPool2d(2, 2),  # 14 * 14

            # 第二个卷积层
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            # 全连接层
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        # 为了更快的训练速度，我们将网络设置成多步模式
        # 根据构造函数的要求，决定是否使用 cupy 后端
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    """
    将图片直接输入到SNN，而不是编码后在输入，是近年来深度SNN的常见做法
    这种情况下，实际的 图片-脉冲 编码是由网络中的前三层
    就是 {Conv2d-BatchNorm2d-IFNode} 完成
    """

    """
    网络的输入直接是 shape=[N, C, H, W] 的图片
    我们将其添加时间维度，并复制 T 次，得到 shape=[T, N, C, H, W] 的序列，然后送入到网络层
    网络的输出定义为最后一层脉冲神经元的脉冲发放频率
    """

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        # 复制T次
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        # 使用网络层
        x_seq = self.conv_fc(x_seq)
        # 生成脉冲结果
        fr = x_seq.mean(0)
        return fr


def main():
    """
    2.指定好训练参数如学习率等以及若干其他配置
    """
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=8, type=int, help='模拟时间步')
    parser.add_argument('-device', default='cuda:0', help='CPU||GPU')
    parser.add_argument('-b', default=32, type=int, help='batch大小')
    parser.add_argument('-epochs', default=20, type=int, metavar='N',
                        help='迭代次数')
    parser.add_argument('-j', default=2, type=int, metavar='N',
                        help='用于数据加载的核数')
    parser.add_argument('-data-dir', type=str, default='fashion-mnist-data', help='MNIST根目录')
    parser.add_argument('-out-dir', type=str, default='./logs_fashion_mnist', help='用于保存的根目录和检查点')
    parser.add_argument('-resume', type=str, help='从check point继续进行实验')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-amp', action='store_true', help='自动混合精度实验')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='使用SGD或者Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='SGD的momentum')
    parser.add_argument('-lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('-tau', default=2.0, type=float, help='LIF神经元参数tau')
    parser.add_argument('-channels', default=64, type=int, help='channels of CSNN')

    # 检查结果
    args = parser.parse_args()
    print(args)
    net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy)
    print(net)
    # 装载至显卡
    net.to(args.device)

    """
    3.加载数据集
    """
    train_dataset = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    test_dataset = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
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

    # 余弦退火算法更新学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

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
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            # 混合精度运算
            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
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

        #保存至数组
        train_losss[epoch]=train_loss
        ar_train_acc[epoch]=train_acc

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        #保存至数组
        test_losss[epoch]=test_loss
        ar_test_acc[epoch]=test_acc

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
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
