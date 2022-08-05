# Pytorch编程笔记

[TOC]

## 1.遇到问题及解决

### 1.1 页面文件太小，无法完成操作

![1659342835394](https://github.com/LinkWithMe/summerHW/blob/main/Week7/image/38)

原因： 模型太大，而系统分配的分页内存太小，无法训练 

解决办法：调整虚拟内存大小，或者更改batch_size大小

将batch_size修改为2，结果成功

### 1.2 cannot import name 'DataLoder' from 'torch.utils.data'

![1659447271918](https://github.com/LinkWithMe/summerHW/tree/main/Week7/image/39)

在导入包时出现这种错误，自己的pytorch版本应该比较新，以前在实验过程中也出现过类似问题

原因：考虑是因为版本问题而导致的导包方式更新

更改导入包方式后成功
