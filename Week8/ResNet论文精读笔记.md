# ResNet论文精读笔记

[TOC]

## 1.摘要

在摘要中，文章首先对提出的创新点进行阐述，并且展示了论文所用方法在许多领域的一些突出成就。

以往的模型设计，往往通过增加网络的深度来实现更好的性能，但问题是，层数堆叠多了，**一是极易造成梯度消失，二是难以训练**，因此先前的模型如果能堆叠到二十多层并成功训练就已经万事大吉了。因此作者提出了残差块的思想。

## 2.图表

首先讲述了论文中的一个图，如下：

![1659861031629](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/1)

随着网络深度的增加，精度达到饱和，然后迅速下降。这不是由过拟合引起的，称之为退化。

第二张图则是残差方法的具体实现：

![1659861170480](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/2)

第三张图是网络架构图：

![1659861195999](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/3)

剩下的则是与先前实验有对应的实验结果图。

## 3.正文

### 3.1 导言

首先说明了，在网络加深的同时，梯度消失和梯度爆炸的问题十分严重，一些处理方法有：

- 权重在随机初始化的时候取更合适的值
- 在中间层加入一些BN，校验输出值、梯度和方差

对于层数很深的网络，其训练误差和测试误差都变得非常大，这不是过拟合。因此作者提出了自己的解决办法。

在导言部分，可以看作是对摘要部分的扩充，解释了residual在做什么事情的同时也进一步地展示了实验结果。**intro是摘要的一个比较扩充的版本，也是一个完整的对整个工作的一个描述**。

### 3.2 Related Work相关工作

**Residual**之前主要用在机器学习和统计学科中，线性模型的解法以及Gradient Boosting算法都是用残差来进行学习的。

**Shortcut**相比于先前的做法而言，更加地纯朴，只使用了简单的加法。

### 3.3 实验实现

首先把短边随机采样到256-480，这样在进行随机切割时，随机性会多一些，并且切割掉像素均值，进行颜色增强。实验中，批量大小为256，学习率是0.1，其他还有一些标准参数，同时没有采用dropout。

在测试时，使用了标准的10个crop testing，就是对每一张测试图片，在里面随机的，或按照一定规律地采样10个图片，在这些子图上做预测，最后结果做平均。同时在很多不同的分辨率上进行采用和测试。

### 3.4 实验

#### 3.4.1 ImageNet

展示了不同层数的残差网络的架构：

![1659864303489](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/4)

不同层数的网络间的不同主要是中间一些残差块的结构不同。

对于表格中的每一个内容：

![1659864475746](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/5)

表示着残差块的内部机构，通道数，以及这个残差块被复制的次数。这些超参数都是网络自动选取以及作者自己设置。

其中表格的最后一行展示了这些网络所做的浮点数运算的数目。

对于如下这个结果图：

![1659864693534](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/6)

数据增强会加大图片的噪音，因此在起步时误差呈现这个关系。同时两次断崖式下跌是因为除以10，也就是乘以0.1操作的原因，并且如果选择过早的进行这步操作，会导致后期收敛无力。

同时，有了残差连接，收敛速度会快很多，后期效果也会更好。

接下来，文章介绍了在输入输出格式不一样的情况下，怎样进行残差连接。有三种方法：

①填零

②投影

③所有的连接都做投影

使用不同方法的结果如下：

![1659864998311](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/7)

可见，BC的结果相差不多，但C的方法会耗费更多的计算资源，因此现在采用的还是B的方式，即在输入输出格式不同的时候，进行投影。

#### 3.4.2 更深层残差网络的实现

在更深层的网络中：

![1659865318473](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/8)

对于256维的输入，首先对其进行一次降维，在降维之后，再用1*1卷积投影返回至原来的维度。虽然右图的通道数是原来的四倍，但是复杂度相较于之前并没有明显变化。

因此再返回至原来的表格中观看：

![1659865479384](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/9)

三层是因为投影的存在。而从34-50的变化，可以说是残差块的个数并没有改变，只是因为每一个块中的更多操作，而导致总层数的变化，所以浮点运算也没有很大改变。

使用不同深度Resnet网络的结果：

![1659881089857](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/10)

而在做了大量的render crop融合之后，效果更好：

![1659881205390](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/11)

#### 3.4.3 CIFAR上的实验结果

最后的实验结果表示，当残差网络的深度达到一定深度后，后面的网络实际上是学习不到有用的东西的。就算是训练了1000层的网络，也可能就前100层的网络有用。

## 4.总结

该论文没有公式所以读起来通俗易懂，但论文没有结论，在日常论文写作中应当注意。

Resnet起作用的原因，在于其对梯度消失问题的解决，这两种方式的求导公式如下：

![1659883664551](https://github.com/LinkWithMe/summerHW/blob/main/Week8/image/12)

通过小数+大数的方式，最后整体上的数值是可以“训练的动”的。

在编写论文时，要尽量在理论+实践两方面讲好。
