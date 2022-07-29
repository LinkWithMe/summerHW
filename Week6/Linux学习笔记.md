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

![1658242263953](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/1)

查询相关资料发现，是由于使用代理软件，winsock出现问题， 可以通过注册表的方式，排除从winsock中排除wsl即可。 因此将如下代码写入reg文件：

```
Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\WinSock2\Parameters\AppId_Catalog\0408F7A3]
"AppFullPath"="C:\\Windows\\System32\\wsl.exe"
"PermittedLspCategories"=dword:80000000
```

执行结果如下：

![1658243604362](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/2)

![1658242485595](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/3)

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

![1658292318256](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/4)

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

![1658471117920](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/xiu_1)

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

###### 错误 cp: missing destination file operand after 'dir3/fun'

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

###### 错误 cp: -r not specified; omitting directory 'dir3/fun'

linux copy命令处理文件复制，因为没有文件名，导致复制了整个目录，需要添加-r参数 ：

```
cbb@LAPTOP-R03NAQL4:~/playground$ cp -r dir3/fun .
cbb@LAPTOP-R03NAQL4:~/playground$ ls -l
total 12
drwxrwxr-x 2 cbb cbb 4096 Jul 22 14:45 dir1
drwxrwxr-x 3 cbb cbb 4096 Jul 22 14:45 dir3
drwxrwxr-x 2 cbb cbb 4096 Jul 22 14:51 fun
```

### 5.使用命令

type 命令是 shell 内部命令，它会显示命令的类别，给出一个特定的命令名（做为参数）。 

```
cbb@LAPTOP-R03NAQL4:/root$ type type
type is a shell builtin
cbb@LAPTOP-R03NAQL4:/root$ type ls
ls is aliased to `ls --color=auto'
cbb@LAPTOP-R03NAQL4:/root$ type cp
cp is /usr/bin/cp
```

有时候在一个操作系统中，不只安装了可执行程序的一个版本。然而在桌面系统中，这并不普遍， 但在大型服务器中，却很平常。为了确定所给定的执行程序的准确位置，使用 which 命令。这个命令只对可执行程序有效，不包括内部命令和命令别名，别名是真正的可执行程序的替代物。 当我们试着使用 shell 内部命令时，例如，cd 命令，我们得不到回应：

```
cbb@LAPTOP-R03NAQL4:/root$ which cd
cbb@LAPTOP-R03NAQL4:/root$
```

--help可以显示用法信息，显示命令所支持的语法和选项说明：

![1658543926395](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/5)

man显示程序手册页，它们不仅仅包括用户命令，也包括系统管理员 命令，程序接口，文件格式等等：

![1658544264912](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/6)

apropos-显示适当的命令，也有可能搜索参考手册列表，基于某个关键字的匹配项。 

whatis-显示非常简洁的命令说明，显示匹配特定关键字的手册页的名字和一行命令说明 

info-显示程序 Info 条目，GNU 项目提供了一个命令程序手册页的替代物，称为”info”。info 内容可通过 info 阅读器 程序读取。info 页是超级链接形式的，和网页很相似。 

##### 5.1 用别名创建自己的命令

联合三个命令同时执行：

```
cbb@LAPTOP-R03NAQL4:~$ cd playground;ls;cd -
dir1  dir3  fun
/home/cbb
```

想将这三个命令结合为一个命令，查询“foo”是否在系统中被占用：

```
cbb@LAPTOP-R03NAQL4:~$ type foo
bash: type: foo: not found
```

创建自己的命令，并执行：

```
cbb@LAPTOP-R03NAQL4:~$ alias foo='cd playground;ls;cd -'
cbb@LAPTOP-R03NAQL4:~$ alias name='foo'
cbb@LAPTOP-R03NAQL4:~$ foo
dir1  dir3  fun
/home/cbb
```

通过type来展示自己的指令详细信息：

```
cbb@LAPTOP-R03NAQL4:~$ type foo
foo is aliased to `cd playground;ls;cd -'
```

通过unalias对指令进行删除：

```
cbb@LAPTOP-R03NAQL4:~$ unalias foo
cbb@LAPTOP-R03NAQL4:~$ type foo
bash: type: foo: not found
```

要查看所有定义在系统环境中的别名，使用不带参数的 alias 命令。 

但是，在命令行中定义别名有点儿小问题。**当你的 shell 会话结束时，别名会消失。**随后的章节里， 我们会了解怎样把自己的别名添加到文件中去，每次我们登录系统，这些文件会建立系统环境。 

### 6.重定向

ls，实际上把他们的运行结果 输送到一个叫做标准输出的特殊文件（经常用 stdout 表示），而它们的状态信息则送到另一个 叫做标准错误的文件（stderr）。许多程序从一个叫做标准输入（stdin）的设备得到输入，默认情况下， 标准输入连接到键盘。 I/O 重定向允许我们可以更改输出走向和输入来向。 

##### 6.1 重定向标准输出和标准错误

重定向输出结果到文件中并检查文件结果：

```
cbb@LAPTOP-R03NAQL4:~$ ls -l playground/dir1 > ls-output.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l ls-output.txt
-rw-rw-r-- 1 cbb cbb 8 Jul 24 10:52 ls-output.txt
```

**删除文件内容，创建新的空文件：**

```
cbb@LAPTOP-R03NAQL4:~$ > ls-output.txt
```

使用”>>“操作符，将导致输出结果添加到文件内容之后 。

因为标准错误和文件描述符2一样，我们用这种 表示法来重定向标准错误： 

捕捉一个命令的所有输出到一个文件：

①传统的方法，在旧版本 shell 中也有效： 

```
cbb@LAPTOP-R03NAQL4:~$ ls -l playground/dir1 > ls-output.txt 2>&1
```

②更精简合理的方法来执行这种联合的重定向：

```
cbb@LAPTOP-R03NAQL4:~$ ls -l playground/dir1 &> ls-output.txt
```

##### 6.2 处理不需要的输出

通过重定向输出结果 到一个特殊的叫做”/dev/null”的文件。这个文件是系统设备，叫做位存储桶，它可以 接受输入，并且对输入不做任何处理。 

```
cbb@LAPTOP-R03NAQL4:~$ ls -l playground/dir1 2> /dev/null
```

##### 6.3 重定向输入

cat 经常被用来显示简短的文本文件。因为 cat 可以 接受不只一个文件作为参数，所以它也可以用来把文件连接在一起。 

通过cat创建一个简短的文本文件，并输出到标准输出文件展示：

```
hello,worldcbb@LAPTOP-R03NAQL4:~$ cat > hello.txt
hello,world
cbb@LAPTOP-R03NAQL4:~$ cat hello.txt
hello,world
```

重定向输入至文件源：

```
cbb@LAPTOP-R03NAQL4:~$ cat < hello.txt
hello,world
```

### 7.从shell角度观察-展开和引用

##### 7.1 一些特殊符号的展开

`*`, 对 shell 来说，有很多的涵义。使这个发生的过程叫做（字符）展开。通过展开， 你输入的字符，在 shell 对它起作用之前，会展开成为别的字符。这种通配符工作机制叫做路径名展开。 

'~'，它用在 一个单词的开头时，它会展开成指定用户的家目录名，如果没有指定用户名，则是当前用户的家目录 

```
cbb@LAPTOP-R03NAQL4:~$ echo ~
/home/cbb
cbb@LAPTOP-R03NAQL4:~$ echo ~cbb
/home/cbb
```

shell 允许算术表达式通过展开来执行。这允许我们把 shell 提示当作计算器来使用，表达式是指算术表达式，它由数值和算术操作符组成。算术表达式只支持整数（全部是数字，不带小数点），但是能执行很多不同的操作。在算术表达式中空格并不重要，并且表达式可以嵌套。 

```
cbb@LAPTOP-R03NAQL4:~$ echo $((2+2))
4
```

花括号展开，可以从一个包含花括号的模式中创建多个文本字符串。

```
cbb@LAPTOP-R03NAQL4:~$ echo begin{A{1,2},B{3,4}}end
beginA1end beginA2end beginB3end beginB4end
```

##### 7.2 参数展开

这个特性在 shell 脚本中比直接在命令行中更有用。它的许多性能 和系统存储小块数据，并给每块数据命名的能力有关系。 许多像这样的小块数据， 更适当些应叫做变量，可以方便地检查它们。 

检查自己的用户名参数：

```
cbb@LAPTOP-R03NAQL4:~$ echo $USER
cbb
```

##### 7.3 控制展开的方式

在使用echo时：

①删除多余的空格

②把“$1” 的值替换为一个空字符串，因为 `1` 是没有定义的变量。

在使用双引号时：

①使用双引号，我们可以阻止单词分割

②在双引号中，参数展开，算术表达式展开，和命令替换仍然有效：

```
cbb@LAPTOP-R03NAQL4:~$ echo "$USER $((2+2)) $(cal)"
cbb 4      July 2022
Su Mo Tu We Th Fr Sa
                1  2
 3  4  5  6  7  8  9
10 11 12 13 14 15 16
17 18 19 20 21 22 23
24 25 26 27 28 29 30
31
```

在使用单引号时：

如果需要禁止所有的展开，我们使用单引号。 

### 8.键盘操作进阶

##### 8.1 自动补全

shell 能帮助你的另一种方式是通过一种叫做自动补全的机制。当你敲入一个命令时， 按下 tab 键，自动补全就会发生。路径名自动补全，这是最常用的形式。 

##### 8.2 历史命令

通过history展示历史命令：

```
...

   84  cat
   85  cat > hello.txt
   86  cat hello.txt
   87  cat > hello.txt
   88  cat hello.txt
   89  cat < hello.txt
   90  hello.txt
   91  ls
   92  echo *s
   93  echo *D
   94  echo *
   95  echo ~
   96  echo ~cbb
   97  echo $((2+2))
   98  echo begin{A{1,2},B{3,4}}end
   99  echo $USER
  100  echo "$USER $((2+2)) $(cal)"
  101  ls ls-output.txt
  102  history
  103  history | less
  104  history
```

通过！和序列使用命令

```
cbb@LAPTOP-R03NAQL4:~$ !102
```

### 9.权限

介绍以下命令：

> - id – 显示用户身份号
> - chmod – 更改文件模式
> - umask – 设置默认的文件权限
> - su – 以另一个用户的身份来运行 shell
> - sudo – 以另一个用户的身份来执行命令
> - chown – 更改文件所有者
> - chgrp – 更改文件组所有权
> - passwd – 更改用户密码

##### 9.1 id

一个用户可能拥有文件和目录。当一个用户拥有一个文件或目录时， 用户对这个文件或目录的访问权限拥有控制权。用户，反过来，又属于一个由一个或多个用户组成的用户组，用户组成员由文件和目录的所有者授予对文件和目录的访问权限。用id命令找到自己的身份信息：

```
cbb@LAPTOP-R03NAQL4:~$ id
uid=1000(cbb) gid=1000(cbb) groups=1000(cbb)
```

当用户创建帐户之后，系统会给用户分配一个号码，叫做用户 ID 或者 uid；

系统又会给这个用户 分配一个原始的组 ID 或者是 gid，这个 gid 可能属于另外的组；

现在的 Linux 会创建一个独一无二的，只有一个成员的用户组，这个用户组与用户同名。这样使某种类型的 权限分配更容易些。 

##### 9.2 cnmod-更改文件格式

写入文件并更改权限：

```
cbb@LAPTOP-R03NAQL4:~$ > foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rw-rw-r-- 1 cbb cbb 0 Jul 24 23:29 foo.txt
cbb@LAPTOP-R03NAQL4:~$ chmod 600 foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rw------- 1 cbb cbb 0 Jul 24 23:29 foo.txt
```

通过传递参数 “600”，我们能够设置文件所有者的权限为读写权限，但通常只会用到一些常见的映射关系： 

7 (rwx)，6 (rw-)，5 (r-x)，4 (r--)，和 0 (---)。 

##### 9.3 umask-设置默认权限

umask 命令控制着文件的默认权限。umask 命令使用八进制表示法来表达从文件模式属性中删除一个位掩码。

```
cbb@LAPTOP-R03NAQL4:~$ rm -f foo.txt
cbb@LAPTOP-R03NAQL4:~$ umask
0002
cbb@LAPTOP-R03NAQL4:~$ > foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rw-rw-r-- 1 cbb cbb 0 Jul 24 23:39 foo.txt
```

这段代码做了以下操作：

①删除foo.txt，以确定从新开始

②运行不带参数的 umask 命令， 看一下当前的掩码值。响应的数值是0002 

③创建文件 foo.txt，并且保留它的权限 

我们可以看到文件所有者和用户组都得到读权限和写权限，而其他人只是得到读权限。 其他人没有得到写权限的原因是由掩码值决定的。重复我们的实验，这次自己设置掩码值： 

```
cbb@LAPTOP-R03NAQL4:~$ rm -f foo.txt
cbb@LAPTOP-R03NAQL4:~$ umask 0000
cbb@LAPTOP-R03NAQL4:~$ > foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rw-rw-rw- 1 cbb cbb 0 Jul 24 23:50 foo.txt
```

当掩码设置为0000（实质上是关掉它）之后，我们看到其他人能够读写文件。

大多数情况下，你不必修改掩码值，系统提供的默认掩码值就很好了。然而，在一些高 安全级别下，你要能控制掩码值。

##### 9.4 su,sudo-更改身份

启动超级用户的 shell：

```
root@LAPTOP-R03NAQL4:~# su -
root@LAPTOP-R03NAQL4:~#
```

企图通过sudo分配超级管理员权限时，出现错误：

```
cbb is not in the sudoers file.  This incident will be reported.
```

切换到root用户进行执行。

更改密码，更改用户组所有权，更改文件所有者和用户组都需要超级用户权限。

### 10.进程

介绍以下命令：

> - ps – 报告当前进程快照
> - top – 显示任务
> - jobs – 列出活跃的任务
> - bg – 把一个任务放到后台执行
> - fg – 把一个任务放到前台执行
> - kill – 给一个进程发送信号
> - killall – 杀死指定名字的进程
> - shutdown – 关机或重启系统

##### 10.1 ps-查看进程

```
cbb@LAPTOP-R03NAQL4:~$ ps
  PID TTY          TIME CMD
  160 pts/0    00:00:00 bash
  167 pts/0    00:00:00 ps
```

上例中，列出了两个进程，进程 160 和进程 167，各自代表命令 bash 和 ps。 

通过ps x展示更多信息：

```
cbb@LAPTOP-R03NAQL4:~$ ps x
  PID TTY      STAT   TIME COMMAND
  160 pts/0    S      0:00 bash
  168 pts/0    R+     0:00 ps x
```

STAT 是 “state” 的简写，它揭示了进程当前状态。

##### 10.2 动态查看进程

虽然 ps 命令能够展示许多计算机运行状态的信息，但是它只是提供，ps 命令执行时刻的机器状态快照。 为了看到更多动态的信息，我们使用 top 命令。top 程序连续显示系统进程更新的信息（默认情况下，每三分钟更新一次） 。

##### 10.3 控制进程

切换回超级管理员下载程序：

遇到错误，无法进行安装

###### 错误：E: Unable to locate package xll-apps

无法找到包，因此使用命令：

```
sudo apt-get update
```

### 11.shell环境

使用以下命令：

> - printenv - 打印部分或所有的环境变量
> - set - 设置 shell 选项
> - export — 导出环境变量，让随后执行的程序知道。
> - alias - 创建命令别名

##### 11.1 printenv-显示环境变量

输入printenv显示所有的环境变量参数

```
cbb@LAPTOP-R03NAQL4:~$ printenv
SHELL=/bin/bash
WSL_DISTRO_NAME=Ubuntu-20.04
NAME=LAPTOP-R03NAQL4
PWD=/home/cbb
LOGNAME=cbb
HOME=/home/cbb
LANG=C.UTF-8
...
```

通过指定代码展示特定变量值：

```
cbb@LAPTOP-R03NAQL4:~$ printenv USER
cbb
```

##### 11.2 set-显示shell和环境变量

当使用没有带选项和参数的 set 命令时，shell 和环境变量二者都会显示，同时也会显示定义的 shell 函数。 

如果 shell 环境中的一个成员既不可用 set 命令也不可用 printenv 命令显示，则这个变量是别名。 输入不带参数的 alias 命令来查看它们。

### 12.VI

vi 很多系统都预装，且是一个轻量级且执行快速的编辑器。

启动vim：

![1658723011630](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/7)

添加新的文件：

```
"foo.txt" [New File]
```

按下i键进入插入模式：

```
-- INSERT --
```

编写号内容后，按下：w进行内容的保存：

```
"foo.txt" [New] 1L, 21C written
```

按下a键来进行行尾插入，按下A键自动移至行尾并且准备行尾插入，**记住按下 Esc 按键来退出插入模式。** 

通过x和d键来进行删除，J把行与行之间连接起来，通过f查询文档

### 13.自定制shell提示符

通过PS1变量来显示我们的用户名，主机名和当前工作目录：

```
cbb@LAPTOP-R03NAQL4:~$ echo $PS1
\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$
```

自定义设计：

首先将PS1中原本的信息保存到另外变量pl1_old中：

```
cbb@LAPTOP-R03NAQL4:~$ ps1_old="$PS1"
cbb@LAPTOP-R03NAQL4:~$ echo $ps1_old
\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$
```

设置为每一次提示符显示都会有提醒音：

```
$ PS1="\a\$ "
```

设置为显示主机名和当天时间信息：

```
$ PS1="\A \h \$ "
16:47 LAPTOP-R03NAQL4 $
```

同时，也可以更改提示符的颜色，修改光标位置。结果可以保存到.bashrc文件中，方便以后加载。

### 14.存储媒介

介绍以下命令：

> - mount – 挂载一个文件系统
> - umount – 卸载一个文件系统
> - fsck – 检查和修复一个文件系统
> - fdisk – 分区表控制器
> - mkfs – 创建文件系统
> - fdformat – 格式化一张软盘
> - dd — 把面向块的数据直接写入设备
> - genisoimage (mkisofs) – 创建一个 ISO 9660的映像文件
> - wodim (cdrecord) – 把数据写入光存储媒介
> - md5sum – 计算 MD5检验码

使用mount 命令来挂载文件系统：

```
cbb@LAPTOP-R03NAQL4:~$ mount
/dev/sdb on / type ext4 (rw,relatime,discard,errors=remount-ro,data=ordered)
none on /mnt/wsl type tmpfs (rw,relatime)
```

这个列表的格式是：设备 on 挂载点 type 文件系统类型(可选的)。

### 15.网络系统

介绍以下命令：

> - ping - 发送 ICMP ECHO_REQUEST 软件包到网络主机
> - traceroute - 打印到一台网络主机的路由数据包
> - netstat - 打印网络连接，路由表，接口统计数据，伪装连接，和多路广播成员
> - ftp - 因特网文件传输程序
> - wget - 非交互式网络下载器
> - ssh - OpenSSH SSH 客户端（远程登录程序）

##### 15.1 ping-检测网络

最基本的网络命令是 ping。这个 ping 命令发送一个特殊的网络数据包，叫做 IMCP ECHO_REQUEST，到 一台指定的主机。大多数接收这个包的网络设备将会回复它，来允许网络连接验证。

尝试验证连接baidu.com的网络情况：

```
cbb@LAPTOP-R03NAQL4:~$ ping baidu.com
PING baidu.com (39.156.66.10) 56(84) bytes of data.
64 bytes from 39.156.66.10: icmp_seq=1 ttl=51 time=22.9 ms
64 bytes from 39.156.66.10: icmp_seq=2 ttl=51 time=25.4 ms
64 bytes from 39.156.66.10: icmp_seq=3 ttl=51 time=25.5 ms
64 bytes from 39.156.66.10: icmp_seq=4 ttl=51 time=25.2 ms
^C
--- baidu.com ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 4134ms
rtt min/avg/max/mdev = 22.884/24.743/25.503/1.077 ms
```

通过Ctrl-c，中断这个命令之后，ping 打印出运行统计信息。一个正常工作的网络会报告 零个数据包丢失。一个成功执行的“ping”命令会意味着网络的各个部件（网卡，电缆，路由，网关） 都处于正常的工作状态。

##### 15.2 traceroute-显示流量列表

traceroute 程序（一些系统使用相似的 tracepath 程序来代替）会显示从本地到指定主机 要经过的所有“跳数”的网络流量列表。同样以百度为例：

```
cbb@LAPTOP-R03NAQL4:~$ tracepath baidu.com
 1?: [LOCALHOST]                      pmtu 1410
 1:  LAPTOP-R03NAQL4.mshome.net                            0.161ms
 1:  LAPTOP-R03NAQL4.mshome.net                            0.148ms
 2:  192.168.243.184                                       1.755ms
 3:  no reply
 4:  172.21.0.98                                          63.105ms
 5:  192.168.65.13                                        63.572ms
 6:  no reply
 7:  111.5.10.45                                          52.290ms
 8:  221.183.74.57                                        70.165ms
 9:  no reply
10:  no reply
11:  39.156.27.5                                          68.389ms
12:  39.156.27.1                                          59.627ms asymm 11
```

到达baidu.com需要12个路由器，自己实验调用的是tracepath方法，输出结果与教程有些许不同。

### 16.查找文件

locate 程序快速搜索路径名数据库，并且输出每个与给定字符串相匹配的文件名。 

##### 16.1 find-查找文件的复杂方式

find 命令的最简单使用是，搜索一个或多个目录。例如，输出我们的家目录列表：

```
cbb@LAPTOP-R03NAQL4:~$ find ~
/home/cbb
/home/cbb/.bash_history
/home/cbb/foo.txt
/home/cbb/.config
/home/cbb/.config/procps
/home/cbb/playground
/home/cbb/playground/dir3
/home/cbb/playground/dir3/fun
/home/cbb/playground/fun
/home/cbb/playground/dir1
/home/cbb/ls-output.txt
/home/cbb/.profile
/home/cbb/.bash_logout
/home/cbb/.viminfo
/home/cbb/.bashrc
/home/cbb/.lesshst
/home/cbb/hello.txt
```

统计文件数量：

```
cbb@LAPTOP-R03NAQL4:~$ find ~ | wc -l
17
```

通过添加文件格式、通配符、操作符的方式来缩小查找范围。

| 操作    | 描述                                                         |
| :------ | :----------------------------------------------------------- |
| -delete | 删除当前匹配的文件。                                         |
| -ls     | 对匹配的文件执行等同的 ls -dils 命令。并将结果发送到标准输出。 |
| -print  | 把匹配文件的全路径名输送到标准输出。如果没有指定其它操作，这是 默认操作。 |
| -quit   | 一旦找到一个匹配，退出。                                     |

### 17.归档和备份

文档压缩程序：

> - gzip – 压缩或者展开文件
> - bzip2 – 块排序文件压缩器

归档程序

> - tar – 磁带打包工具
> - zip – 打包和压缩文件

文档同步程序：

> - rsync – 同步远端文件和目录
>

##### 17.1 gzip-压缩和解压文件

当执行 gzip 命令时，则原始文件的压缩版会替代原始文件。 相对应的 gunzip 程序被用来把压缩文件复原为没有被压缩的版本：

```
cbb@LAPTOP-R03NAQL4:~$ ls -l /etc > foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.*
-rw-rw-r-- 1 cbb cbb 10310 Jul 25 17:57 foo.txt
cbb@LAPTOP-R03NAQL4:~$ gzip foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.*
-rw-rw-r-- 1 cbb cbb 1922 Jul 25 17:57 foo.txt.gz
cbb@LAPTOP-R03NAQL4:~$ gunzip foo.txt.gz
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.*
-rw-rw-r-- 1 cbb cbb 10310 Jul 25 17:57 foo.txt
```

我们运行 gzip 命令，它会把原始文件替换为一个叫做 foo.txt.gz 的压缩文件。在 foo.* 文件列表中，我们看到原始文件已经被压缩文件替代了，并将这个压缩文件大约是原始 文件的十五分之一。接下来，我们运行 gunzip 程序来解压缩文件。随后，我们能见到压缩文件已经被原始文件替代了， 同样地保留了相同的权限和时间戳。 

 gunzip 程序，会解压缩 gzip 文件，假定那些文件名的扩展名是.gz，所以没有必要指定它， 只要指定的名字与现有的未压缩文件不冲突就可以。

##### 17.2 bzip2-更高的压缩级别

舍弃了压缩速度，而实现了更高的压缩级别。在大多数情况下，它的工作模式等同于 gzip。 由 bzip2 压缩的文件，用扩展名 .bz2 来表示。

```
cbb@LAPTOP-R03NAQL4:~$ ls -l /etc > foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rw-rw-r-- 1 cbb cbb 10310 Jul 25 22:11 foo.txt
cbb@LAPTOP-R03NAQL4:~$ bzip2 foo.txt
cbb@LAPTOP-R03NAQL4:~$ bzip2 foo.txt.bz2
bzip2: Input file foo.txt.bz2 already has .bz2 suffix.
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt.bz2
-rw-rw-r-- 1 cbb cbb 1800 Jul 25 22:11 foo.txt.bz2
```

##### 17.3 tar-归档文件

一个常见的，与文件压缩结合一块使用的文件管理任务是归档。归档就是收集许多文件，并把它们捆绑成一个大文件的过程。归档经常作为系统备份的一部分来使用。当把旧数据从一个系统移到某 种类型的长期存储设备中时，也会用到归档程序。我们经常看到扩展名为 .tar 或者 .tgz 的文件，它们各自表示“普通” 的 tar 包和被 gzip 程序压缩过的 tar 包。

通过tar压缩playground，其包含整个 playground 目录层次结果。 

```
cbb@LAPTOP-R03NAQL4:~$ tar cf playground.tar playground
```

显示文件的内容， 添加选项 v 显示详细信息：

```
cbb@LAPTOP-R03NAQL4:~$ tar tf playground.tar
playground/
playground/dir3/
playground/dir3/fun/
playground/fun/
playground/dir1/
cbb@LAPTOP-R03NAQL4:~$ tar tvf playground.tar
drwxrwxr-x cbb/cbb           0 2022-07-22 14:51 playground/
drwxrwxr-x cbb/cbb           0 2022-07-22 14:45 playground/dir3/
drwxrwxr-x cbb/cbb           0 2022-07-22 14:42 playground/dir3/fun/
drwxrwxr-x cbb/cbb           0 2022-07-22 14:51 playground/fun/
drwxrwxr-x cbb/cbb           0 2022-07-22 14:45 playground/dir1/
```

抽取 tar 包 playground 到一个新位置。我们先创建一个名为 foo 的新目录，更改目录， 然后抽取 tar 包中的文件： 

```
cbb@LAPTOP-R03NAQL4:~$ mkdir foo
cbb@LAPTOP-R03NAQL4:~$ cd foo
cbb@LAPTOP-R03NAQL4:~/foo$ tar xf ../playground.tar
cbb@LAPTOP-R03NAQL4:~/foo$ ls
playground
```

这个归档文件已经被成功地安装了，就是创建了一个精确的原始文件的副本。

使用 tar 和 find 命令，来创建逐渐增加的目录树或者整个系统的备份，是个不错的方法。通过 find 命令匹配新于某个时间戳的文件，我们就能够创建一个归档文件，其只包含新于上一个 tar 包的文件， 假定这个时间戳文件恰好在每个归档文件创建之后被更新了。 

##### 17.4 zip-压缩和打包工具

通常的zip唤醒命令如下:

```
zip options zipfile file...
```



制作一个 playground 的 zip 版本的文件包，这样做： 

```
cbb@LAPTOP-R03NAQL4:~$ zip -r playground.zip playground
```

除非我们包含-r 选项，要不然只有 playground 目录（没有任何它的内容）被存储。 

##### 17.5 rsync-同步文件和目录

维护系统备份的常见策略是保持一个或多个目录与另一个本地系统（通常是某种可移动的存储设备） 或者远端系统中的目录（或多个目录）同步。通过使用 rsync 远端更新协议，此协议 允许 rsync 快速地检测两个目录的差异，执行最小量的复制来达到目录间的同步。比起其它种类的复制程序， 这就使 rsync 命令非常快速和高效。

一般的唤醒语法：

```
 rsync options source destination
```

这里 source 和 destination 是下列选项之一：

- 一个本地文件或目录
- 一个远端文件或目录，以[user@]host:path 的形式存在
- 一个远端 rsync 服务器，由 rsync://[user@]host[:port]/path 指定

同步 playground 目录和它在 foo 目录中相对应的副本：

```
cbb@LAPTOP-R03NAQL4:~$ rm -rf foo/*
cbb@LAPTOP-R03NAQL4:~$ rsync -av playground foo
sending incremental file list
playground/
playground/dir1/
playground/dir3/
playground/dir3/fun/
playground/fun/

sent 191 bytes  received 36 bytes  454.00 bytes/sec
total size is 0  speedup is 0.00
```

包括了-a 选项（递归和保护文件属性）和-v 选项（冗余输出），来在 foo 目录中制作一个 playground 目录的镜像。最后两行的总结信息说明了复制的数量。

### 18.正则表达式

正则表达式是一种符号表示法，被用来识别文本模式。在某种程度上，它们与匹配文件和路径名的 shell 通配符比较相似，但其规模更庞大。 

##### 18.1 grep程序

 grep 程序以这样的方式来接受选项和参数： 

```
grep [options] regex [file...]
```

创建用于实验的文件：

```
cbb@LAPTOP-R03NAQL4:~$ ls /bin > dirlist-bin.txt
cbb@LAPTOP-R03NAQL4:~$ ls /user/bin > dirlist-usr-bin.txt
ls: cannot access '/user/bin': No such file or directory
cbb@LAPTOP-R03NAQL4:~$ ls /usr/bin > dirlist-usr-bin.txt
cbb@LAPTOP-R03NAQL4:~$ ls /sbin > dirlist-sbin.txt
cbb@LAPTOP-R03NAQL4:~$ ls /usr/sbin > dirlist-usr-sbin.txt
cbb@LAPTOP-R03NAQL4:~$ ls dirlist*.txt
dirlist-bin.txt  dirlist-sbin.txt  dirlist-usr-bin.txt  dirlist-usr-sbin.txt
```

用正则式进行简单搜索，grep程序在所有列出的文件中搜索字符串 bzip：

```
cbb@LAPTOP-R03NAQL4:~$ grep bzip dirlist*.txt
dirlist-bin.txt:bzip2
dirlist-bin.txt:bzip2recover
dirlist-usr-bin.txt:bzip2
dirlist-usr-bin.txt:bzip2recover
```

这个正则表达式“bzip”意味着，匹配项所在行至少包含4个字符，并且按照字符 “b”, “z”, “i”, 和 “p”的顺序 出现在匹配行的某处，字符之间没有其它的字符。字符串“bzip”中的所有字符都是原义字符，因为 它们匹配本身。

除了原义字符之外，正则表达式也可能包含元字符，其被用来指定更复杂的匹配项。 

第一个元字符是圆点字符，其被用来匹配任意字符。如果我们在正则表达式中包含它， 它将会匹配在此位置的任意一个字符：

```
cbb@LAPTOP-R03NAQL4:~$ grep -h '.zip' dirlist*.txt
bunzip2
bzip2
bzip2recover
gpg-zip
gunzip
gzip
bunzip2
bzip2
bzip2recover
gpg-zip
gunzip
gzip
```

在正则表达式中，插入符号和美元符号被看作是锚点。这意味着正则表达式 只有在文本行的开头或末尾被找到时，才算发生一次匹配。

通过中括号表达式，我们能够指定 一个字符集合（包含在不加中括号的情况下会被解释为元字符的字符）来被匹配。在这个例子里，使用了一个两个字符的集合，我们匹配包含字符串“bzip”或者“gzip”的任意行 ： 

```
cbb@LAPTOP-R03NAQL4:~$ grep -h '[bg]zip' dirlist*.txt
bzip2
bzip2recover
gzip
bzip2
bzip2recover
gzip
```

否定方法，得到一个文件列表，它们的文件名都包含字符串“zip”，并且“zip”的前一个字符 是除了“b”和“g”之外的任意字符：

```
cbb@LAPTOP-R03NAQL4:~$ grep -h '[^bg]zip' dirlist*.txt
```

通过字符来进行筛选，首字符是大写的A-Z：

```
cbb@LAPTOP-R03NAQL4:~$ grep -h '^[A-Z]' dirlist*.txt
NF
VGAuthService
X11
NF
VGAuthService
X11
```

##### 18.2 POSIX字符集

POSIX 标准介绍了一种叫做 locale 的概念，其可以被调整，来为某个特殊的区域， 选择所需的字符集。通过使用下面这个命令，我们能够查看到我们系统的语言设置： 

```
cbb@LAPTOP-R03NAQL4:~$ echo $LANG
C.UTF-8
```

### 19.文本处理

介绍以下命令：

> - cat – 连接文件并且打印到标准输出
> - sort – 给文本行排序
> - uniq – 报告或者省略重复行
> - cut – 从每行中删除文本区域
> - paste – 合并文件文本行
> - join – 基于某个共享字段来联合两个文件的文本行
> - comm – 逐行比较两个有序的文件
> - diff – 逐行比较文件
> - patch – 给原始文件打补丁
> - tr – 翻译或删除字符
> - sed – 用于筛选和转换文本的流编辑器
> - aspell – 交互式拼写检查器

##### 19.1 cat

通过cat，-A 选项， 其用来在文本中显示非打印字符；-n，其给文本行添加行号 ； -s ；禁止输出多个空白行 

##### 19.2 sort

 sort 程序对标准输入的内容，或命令行中指定的一个或多个文件进行排序，然后把排序结果发送到标准输出。使用与 cat 命令相同的技巧，我们能够演示如何用 sort 程序来处理标准输入： 

```
cbb@LAPTOP-R03NAQL4:~$ sort >foo.txt
c
b
a
cbb@LAPTOP-R03NAQL4:~$ cat foo.txt
a
b
c
```

因为 sort 程序能接受命令行中的多个文件作为参数，所以有可能把多个文件合并成一个有序的文件。

##### 19.3 uniq

当给定一个 排好序的文件（包括标准输出），uniq 会删除任意重复行，并且把结果发送到标准输出。 它常常和 sort 程序一块使用，来清理重复的输出。 

```
cbb@LAPTOP-R03NAQL4:~$ cat > foo.txt
a
a
c
c
d
cbb@LAPTOP-R03NAQL4:~$ uniq foo.txt
a
c
d
```

##### 19.4 切片与复制

cut 程序被用来从文本行中抽取文本，并把其输出到标准输出。它能够接受多个文件参数或者标准输入。 

paste 命令的功能正好与 cut 相反。它会添加一个或多个文本列到文件中，而不是从文件中抽取文本列。 它通过读取多个文件，然后把每个文件中的字段整合成单个文本流，输入到标准输出。 

join 命令类似于 paste，它会往文件中添加列，但是它使用了独特的方法来完成。 一个 join 操作通常与关系型数据库有关联，在关系型数据库中来自多个享有共同关键域的表格的 数据结合起来，得到一个期望的结果。这个 join 程序执行相同的操作。它把来自于多个基于共享 关键域的文件的数据结合起来。 

##### 19.5 cut-比较文本

 一名系统管理员可能，例如，需要拿现有的配置文件与先前的版本做比较，来诊断一个系统错误。 同样的，一名程序员经常需要查看程序的修改。

comm 程序会比较两个文本文件，并且会显示每个文件特有的文本行和共有的文本行，比较结果如下：

```
cbb@LAPTOP-R03NAQL4:~$ comm file1.txt file2.txt
a
                b
                c
                d
        e
```

comm 命令产生了三列输出。第一列包含第一个文件独有的文本行；第二列， 文本行是第二列独有的；第三列包含两个文件共有的文本行。comm 支持 -n 形式的选项，这里 n 代表 1，2 或 3。这些选项使用的时候，指定了要隐藏的列。 

##### 19.6 diff

类似于 comm 程序，diff 程序被用来监测文件之间的差异。然而，diff 是一款更加复杂的工具，它支持 许多输出格式，并且一次能处理许多文本文件。软件开发员经常使用 diff 程序来检查不同程序源码 版本之间的更改，diff 能够递归地检查源码目录，经常称之为源码树。diff 程序的一个常见用例是 创建 diff 文件或者补丁，它会被其它程序使用，例如 patch 程序，来把文件 从一个版本转换为另一个版本。 

```
cbb@LAPTOP-R03NAQL4:~$ diff  file1.txt file2.txt
1d0
< a
4a4
> e
```

对两个文件之间差异的简短描述。在默认格式中， 每组的更改之前都是一个更改命令，其形式为 range operation range ， 用来描述要求更改的位置和类型，从而把第一个文件转变为第二个文件 。

通过-u来指定更加简洁的上下文模式：

```
cbb@LAPTOP-R03NAQL4:~$ diff -u  file1.txt file2.txt
--- file1.txt   2022-07-26 18:20:14.466640300 +0800
+++ file2.txt   2022-07-26 18:20:26.246640300 +0800
@@ -1,4 +1,4 @@
-a
 b
 c
 d
+e
```

上下文模式和统一模式之间最显著的差异就是重复上下文的消除，这就使得统一模式的输出结果要比上下文 模式的输出结果简短。 

##### 19.7 patch

patch 程序被用来把更改应用到文本文件中。它接受从 diff 程序的输出，并且通常被用来 把较老的文件版本转变为较新的文件版本。使用 diff/patch 组合提供了 两个重大优点：

1. 一个 diff 文件非常小，与整个源码树的大小相比较而言。
2. 一个 diff 文件简洁地显示了所做的修改，从而允许程序补丁的审阅者能快速地评估它。

建议使用diff命令的方式：

```
diff -Naur old_file new_file > diff_file
```

创建了一个名为 patchfile.txt 的 diff 文件，然后使用 patch 程序， 来应用这个补丁。注意我们没有必要指定一个要修补的目标文件，因为 diff 文件（在统一模式中）已经 在标题行中包含了文件名。一旦应用了补丁，我们能看到，现在 file1.txt 与 file2.txt 文件相匹配了。 

##### 19.8 运行时编辑

tr

这个 tr 程序被用来更改字符。我们可以把它看作是一种基于字符的查找和替换操作。 换字是一种把字符从一个字母转换为另一个字母的过程。例如，把小写字母转换成大写字母就是 换字。我们可以通过 tr 命令来执行这样的转换，如下所示：

```
cbb@LAPTOP-R03NAQL4:~$ echo "hello,world" | tr a-z A-Z
HELLO,WORLD
```

sed

这个命令认为是相似于 vi 中的“替换” （查找和替代）命令。sed 中的命令开始于单个字符。在上面的例子中，这个替换命令由字母 s 来代表，其后跟着查找 和替代字符串，斜杠字符做为分隔符。且s是最常用的编辑命令。

```
cbb@LAPTOP-R03NAQL4:~$ echo "hello" | sed 's/hello/bye/'
bye
```

aspell

虽然 aspell 程序大多被其它需要拼写检查能力的 程序使用，但它也可以作为一个独立的命令行工具使用。它能够智能地检查各种类型的文本文件， 包括 HTML 文件，C/C++ 程序，电子邮件和其它种类的专业文本。 

### 20.格式化输出

命令如下：

> - nl – 添加行号
> - fold – 限制文件列宽
> - fmt – 一个简单的文本格式转换器
> - pr – 让文本为打印做好准备
> - printf – 格式化数据并打印出来
> - groff – 一个文件格式系统

nl：

```
cbb@LAPTOP-R03NAQL4:~$ nl foo.txt
     1  a
     2  a
     3  c
     4  c
     5  d
```

fold

折叠是将文本的行限制到特定的宽的过程。像我们的其他命令，fold 接受一个或多个文件及标准输入：

```
cbb@LAPTOP-R03NAQL4:~$ echo "hello,world" | fold -w 4
hell
o,wo
rld
```

fmt：

fmt 程序同样折叠文本，外加很多功能。它接受文本或标准输入并且在文本流上呈现照片转换。基础来说，他填补并且将文本粘帖在 一起并且保留了空白符和缩进。 

### 21.编写shell脚本

一个 shell 脚本就是一个包含一系列命令的文件。shell 读取这个文件，然后执行 文件中的所有命令，就好像这些命令已经直接被输入到了命令行中一样。编写过程：

①创建文本：

```
cbb@LAPTOP-R03NAQL4:~$ cat > foo.txt
echo "hello,my name is cbb"
```

②更改可执行权限：

```
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rw-rw-r-- 1 cbb cbb 10 Jul 26 18:08 foo.txt
cbb@LAPTOP-R03NAQL4:~$ chmod 755 foo
cbb@LAPTOP-R03NAQL4:~$ chmod 755 foo.txt
cbb@LAPTOP-R03NAQL4:~$ ls -l foo.txt
-rwxr-xr-x 1 cbb cbb 10 Jul 26 18:08 foo.txt
```

对于脚本文件，有两个常见的权限设置；权限为755的脚本，则每个人都能执行，和权限为700的 脚本，只有文件所有者能够执行。 

③执行脚本：

```
cbb@LAPTOP-R03NAQL4:~$ ./foo.txt
hello,my name is cbb
```

通过使用行继续符（反斜杠-回车符序列）和缩进，这个复杂命令的逻辑性更清楚地描述给读者。 这个技巧在命令行中同样生效，虽然很少使用它，因为输入和编辑这个命令非常麻烦。脚本和 命令行的一个区别是，脚本可能雇佣 tab 字符拉实现缩进，然而命令行却不能，因为 tab 字符被用来 激活自动补全功能。 

## 参考资料

[1] [一劳永逸，wsl2出现“参考的对象类型不支持尝试的操作”的解决办法_桑榆肖物的博客-CSDN博客_参考的对象类型不支持尝试的操作](https://blog.csdn.net/marin1993/article/details/119841299) 

[2] [Windows11下安装Linux 操作系统 | 使用WSL2安装Ubuntu | Windows10下安装Linux_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Qb4y1a7KZ?spm_id_from=333.337.search-card.all.click&vd_source=2e2e5482b016884769e4190397c8bfbb) 

[3] [linux创建新的用户_xiaoweiwei99的博客-CSDN博客_linux添加用户](https://blog.csdn.net/qq_46416934/article/details/123973715) 

[4] [Linux新手入门：Unable to locate package错误解决办法_ljf_study的博客-CSDN博客](https://blog.csdn.net/ljf_study/article/details/81591059) 
