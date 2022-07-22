# Linux学习笔记

[TOC]

## 一、Linux的配置与安装

### 1.安装过程中碰到的问题

##### 1.1 网络问题

原本应该是同时安装好wsl+Ubuntu，但是安装Ubuntn时，断网了一次，Ubuntn安装失败，因此上网搜索有两种解决方式：

①通过w11自带的store商店下载Ubuntn

②在powershell中通过指令下载

通过②方式完成下载

##### 1.2 参考的对象类型不支持尝试的操作

运行Ubuntn，出现以下错误，无法响应指令：

![1658242263953](https://github.com/LinkWithMe/summerHW/blob/main/Week5/image_week5/46)

查询相关资料发现，是由于使用代理软件，winsock出现问题， 可以通过注册表的方式，排除从winsock中排除wsl即可。 因此将如下代码写入reg文件：

```
Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinSock2\Parameters\AppId_Catalog\0408F7A3]
"AppFullPath"="C:\\Windows\\System32\\wsl.exe"
"PermittedLspCategories"=dword:80000000
```

执行结果如下：

![1658243604362](https://github.com/LinkWithMe/summerHW/blob/main/Week5/image_week5/47)

![1658242485595](https://github.com/LinkWithMe/summerHW/blob/main/Week5/image_week5/48)

成功完成安装，并且通过以下代码检查了网络连接、git与Python：

```
/sys# ip addr

/sys# git

/sys# python3
```

结果与所参考的视频资料一致无误。

##### 1.3 创建新用户

由于1.2的错误，导致没有创建自己的linux用户，因此上述的编写都是通过 ~# 来进行操作的。

创建新用户：

![1658292318256](https://github.com/LinkWithMe/summerHW/blob/main/Week5/image_week5/49)

## 二、Linux学习

### 1.基础知识介绍

 “Linux”指的是内核以及在一个典型的 Linux 发行版中所包含的所有免费及开源软件； 也就是说，整个 Linux 生态系统，不只有 GNU 项目软件。 

### 2.shell

我的终端仿真器运行起始如下：

```
cbb@LAPTOP-R03NAQL4:/root$
```

它通常包括你的用户名@主机名，紧接着当前工作目录（稍后会有更多介绍）和一个美元符号。如果提示符的最后一个字符是“#”, 而不是“$”, 那么这个终端会话就有超级用户权限。 这意味着，我们或者是以 root 用户的身份登录，或者是我们选择的终端仿真器提供超级用户（管理员）权限。 

不要在一个终端窗口里使用 Ctrl-c 和 Ctrl-v 快捷键来执行拷贝和粘贴操作。 它们不起作用。对于 shell 来说，这两个控制代码有着不同的含义，它们在早于 Microsoft Windows （定义复制粘贴的含义）许多年之前就赋予了不同的意义。 

尝试的运行命令：

```
date 

cal  

df

exit
```

### 3.文件系统中跳转

类似于 Windows，一个“类 Unix” 的操作系统，比如说 Linux，以分层目录结构来组织所有文件。 这就意味着所有文件组成了一棵树型目录（有时候在其它系统中叫做文件夹）， 这个目录树可能包含文件和其它的目录。文件系统中的第一级目录称为根目录。 根目录包含文件和子目录，子目录包含更多的文件和子目录，依此类推。 

文件命名规则：

以 “.” 字符开头的文件名是隐藏文件。这仅表示，ls 命令不能列出它们， 用 ls -a 命令就可以了。

文件名和命令名是大小写敏感的。文件名 “File1” 和 “file1” 是指两个不同的文件名。

最重要的是，不要在文件名中使用空格。如果你想表示词与 词间的空格，用下划线字符来代替。 

### 4.操作文件和目录

借助通配符， 为文件名构建非常复杂的选择标准成为可能。下面是一些类型匹配的范例: 

![1658471117920](https://github.com/LinkWithMe/summerHW/blob/main/Week5/image_week5/50)

进行新建目录和目录移动操作：

```
cbb@LAPTOP-R03NAQL4:/root$ cd
cbb@LAPTOP-R03NAQL4:~$ mkdir playground
cbb@LAPTOP-R03NAQL4:~$ cd playground
cbb@LAPTOP-R03NAQL4:~/playground$ mkdir dir1 dir2
cbb@LAPTOP-R03NAQL4:~/playground$ cp /etc/passwd
cp: missing destination file operand after '/etc/passwd'
Try 'cp --help' for more information.
cbb@LAPTOP-R03NAQL4:~/playground$ ls -l
total 8
drwxrwxr-x 2 cbb cbb 4096 Jul 22 14:42 dir1
drwxrwxr-x 2 cbb cbb 4096 Jul 22 14:42 dir2
cbb@LAPTOP-R03NAQL4:~/playground$ mv dir2 fun
cbb@LAPTOP-R03NAQL4:~/playground$ mv fun dir1
cbb@LAPTOP-R03NAQL4:~/playground$ mkdir dir3
cbb@LAPTOP-R03NAQL4:~/playground$ mv dir1/fun dir3
cbb@LAPTOP-R03NAQL4:~/playground$ mv dir1/fun
```

尝试对目录进行拷贝：

##### 4.1 错误 cp: missing destination file operand after 'dir3/fun'

丢失目标文件操作数，在路径后面加上“.”就好了。因为“.”代表当前路径，所以没有加“.”，就代表当前目录，就没有操作的目标，因此就会提示错误。 

```
cbb@LAPTOP-R03NAQL4:~/playground$ cp /dir3/fun
cp: missing destination file operand after '/dir3/fun'
Try 'cp --help' for more information.
cbb@LAPTOP-R03NAQL4:~/playground$ cp /dir3/fun .
cp: cannot stat '/dir3/fun': No such file or directory
cbb@LAPTOP-R03NAQL4:~/playground$ cp dir3/fun .
cp: -r not specified; omitting directory 'dir3/fun'
```

##### 4.2 错误 cp: -r not specified; omitting directory 'dir3/fun'

linux copy命令处理文件复制，因为没有文件名，导致复制了整个目录，需要添加-r参数 ：

```
cbb@LAPTOP-R03NAQL4:~/playground$ cp -r dir3/fun .
cbb@LAPTOP-R03NAQL4:~/playground$ ls -l
total 12
drwxrwxr-x 2 cbb cbb 4096 Jul 22 14:45 dir1
drwxrwxr-x 3 cbb cbb 4096 Jul 22 14:45 dir3
drwxrwxr-x 2 cbb cbb 4096 Jul 22 14:51 fun
```

## 参考资料

[1] [一劳永逸，wsl2出现“参考的对象类型不支持尝试的操作”的解决办法_桑榆肖物的博客-CSDN博客_参考的对象类型不支持尝试的操作](https://blog.csdn.net/marin1993/article/details/119841299) 

[2] [Windows11下安装Linux 操作系统 | 使用WSL2安装Ubuntu | Windows10下安装Linux_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Qb4y1a7KZ?spm_id_from=333.337.search-card.all.click&vd_source=2e2e5482b016884769e4190397c8bfbb) 

[3] [linux创建新的用户_xiaoweiwei99的博客-CSDN博客_linux添加用户](https://blog.csdn.net/qq_46416934/article/details/123973715) 
