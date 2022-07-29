# Conda Jupyter安装及学习

[TOC]

### 1.Miniconda安装

##### 1.1 下载Miniconda

使用命令如下：

```
root@LAPTOP-R03NAQL4:~# wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

下载成功后的提醒：

![1658897960008](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/8)

使用如下命令进行进一步安装：

```
root@LAPTOP-R03NAQL4:~# bash Miniconda3-latest-Linux-x86_64.sh
```

![1658908214150](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/9)

安装至如下目录：

![1658908264307](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/10)

安装完毕：

```
conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

配置环境变量：

```
root@LAPTOP-R03NAQL4:~# conda -v
conda: command not found
root@LAPTOP-R03NAQL4:~# export PATH=/root/miniconda3/bin/:$PATH
root@LAPTOP-R03NAQL4:~# conda -V
conda 4.12.0
```

##### 1.2 使用conda创建新环境

配置python：

```
 (base) root@LAPTOP-R03NAQL4:~# conda create -n envName python=3.9.12
```

通过conda list检查安装的包：

![1658910417677](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/11)

检查自己的CUDA版本：

![1658919057005](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/12)

安装pytorch命令：

```
(base) root@LAPTOP-R03NAQL4:~# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

安装完毕：

###### ![1658926559743](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/13)

验证：

输入如下命令：

```
(base) root@LAPTOP-R03NAQL4:~# python
Python 3.9.12 (main, Apr  5 2022, 06:56:58)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.

import torch
```

无错误报出，证明安装完毕！

并且可以使用自己的GPU1650ti

![1658926686753](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/14)

### 2.Jupyter安装

##### 2.1安装jupyter notebook

执行命令：

```
(base) root@LAPTOP-R03NAQL4:~# pip install Jupyter -i https://pypi.doubanio.com/simple
```

##### 2.2 环境配置

下载完毕后，生成配置文件:

```
(base) root@LAPTOP-R03NAQL4:~# jupyter-notebook --generate-config
Writing default config to: /root/.jupyter/jupyter_notebook_config.py
```

找到文中内容并进行修改：

![1658939113017](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/15)

更新文件：

```
(base) root@LAPTOP-R03NAQL4:~# cat >> ~/.bashrc
export BROWSER='/mnt/c/Program Files(x86)/Microsoft/Edge/Application.msedge.exe'
(base) root@LAPTOP-R03NAQL4:~# source ~/.bashrc
```

### 3.Jupyter Notebook学习

##### 3.1 简介

Jupyter Notebook是以网页的形式打开，可以在网页页面中**直接编写代码**和**运行代码**，代码的**运行结果**也会直接在代码块下显示的程序。如在编程过程中需要编写说明文档，可在同一个页面中直接编写，便于作及时的说明和解释。 

##### 3.2 设置文件存放位置

当执行完启动命令之后，浏览器将会进入到Notebook的主页面，如下图所示。 

![1658978624266](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/16)

修改文件存储位置，首先创建文件夹：

```
(base) root@LAPTOP-R03NAQL4:~# mkdir mynetbookdir
(base) root@LAPTOP-R03NAQL4:~# cd mynetbookdir
(base) root@LAPTOP-R03NAQL4:~/mynetbookdir# pwd
/root/mynetbookdir
```

更改配置文件后，重新打开jupyter，发现见面发生改变：

![1658979313435](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/17)

##### 3.3 页面基本操作

Files页面是用于管理和创建文件相关的类目。

对于现有的文件，可以通过勾选文件的方式，对选中文件进行复制、重命名、移动、下载、查看、编辑和删除的操作。

同时，也可以根据需要，在“New”下拉列表中选择想要创建文件的环境，进行创建“ipynb”格式的笔记本、“txt”格式的文档、终端或文件夹。

##### 3.4 关联conda

使用命令如下：

```
(base) root@LAPTOP-R03NAQL4:~# conda install nb_conda
```

安装完毕界面：

![1658980454768](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/18)

出现conda选项：

![1658980493987](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/19)

安装Markdown生成目录：

```
(base) root@LAPTOP-R03NAQL4:~# conda install -c conda-forge jupyter_contrib_nbextensions
```

安装完毕：

![1658980871169](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/1658980871169.png)

##### 3.4 基本操作

通过如下代码粘贴网络地址：

```text
%load URL
```

![1658997954554](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/1658997954554.png)

加载本地Python文件：

```
%load Python文件的绝对路径
```

输入命令后，可以按`CTRL 回车`来执行命令。第一次执行，是将本地的Python文件内容加载到单元格内。此时，Jupyter Notebook会自动将“%load”命令注释掉（即在前边加井号“#”），以便在执行已加载的文件代码时不重复执行该命令；第二次执行，则是执行已加载文件的代码。 

通过如下命令来获取当前所在位置的绝对路径

```
%pwd

!pwd
```

![1658998365026](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/22)

注意，其中!pwd属于shell语法，**即在Jupyter Notebook中执行shell命令的语法** 。

### 4.问题及解决

##### 4.1 “E: UNABLE TO LOCATE PACKAGE”解决办法

在学习过程中，一直遇到如上所示无法定位到包的问题，查阅相关资料发现，可能是因为源的问题，导致不能够连接到源，导致不能下载包，因此切换到阿里云的源进行尝试操作：

①备份源：

```
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak 
```

②换源：

进入配置文件：

```
vi /etc/apt/sources.list
```

删除原来的数据之后，添加新的数据如下所示：

![1658853044707](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/23)

③更新列表

```
sudo apt-get update

sudo apt-get upgrade
```

④完成安装：

```
root@LAPTOP-R03NAQL4:~# apt-get install wget
Reading package lists... Done
Building dependency tree
Reading state information... Done
wget is already the newest version (1.20.3-1ubuntu2).
wget set to manually installed.
0 upgraded, 0 newly installed, 0 to remove and 7 not upgraded.
```

##### 4.2 错误“Running as root is not recommended. Use --allow-root to bypass.”

在运行jupyter notebook遇到如上错误，找到配置文件，修改内容为True并去掉注释

![1658940745629](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/24)

但是发现依旧报错，不进行文件的更改，而是使用如下命令：

```
(base) root@LAPTOP-R03NAQL4:~# jupyter notebook --allow-root
```

运行结果如下：

![1658975743986](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/25)

粘贴其中的URLs到浏览器中，运行成功：

![1658975941083](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/26)

### 5.存在问题

在学习过程中，起初在base环境中已经基本完成了整体安装过程，但是在最后一步，安装这个插件的时候：

```text
conda install -c conda-forge jupyter_contrib_nbextensions
```

安装后，一直报错

![1658999106821](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/27)

一直无法debug，最后卸载了jupyter，但是没卸载干净导致重装也无法运行，并且在环境中产生了大量无用的包。最后在新环境中完成实验。但是base环境中那么多包仍是一个问题。之后有时间要把这些包uninstall一下。

尝试使用如下代码进行清除：

```
(base) root@LAPTOP-R03NAQL4:~# conda clean -a
```

清除结果如下：

![1659013555688](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/28)

尝试对base环境进行回滚：

![1659023377190](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/29)

但是出现缺失包的错误：

![1659023414275](https://github.com/LinkWithMe/summerHW/blob/main/Week6/image/30)

### 参考资料

[1] [windows下如何复制文字到ubuntu上 - CSDN](https://www.csdn.net/tags/NtTagg2sMjkxMDQtYmxvZwO0O0OO0O0O.html) 

[2]  [Linux Ubuntu 修改 /etc/apt/sources.list (镜像源)文件和ifconfig命令及net-tools与vim问题解决(非常实用)_书启秋枫的博客-CSDN博客_/etc/apt/sources.list如何修改](https://blog.csdn.net/qq_45037155/article/details/123361839) 

[3] [windows系统下装载wsl2，安装Miniconda3或Anaconda进行生信准备工作_Bio大恐龙的博客-CSDN博客_wsl安装miniconda](https://blog.csdn.net/ouyangk1026/article/details/125192219) 

[4] [本地复制文本无法在Ubuntu中粘贴问题_BanFS的博客-CSDN博客_ubuntu无法复制粘贴](https://blog.csdn.net/banfushen007/article/details/104246719?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-2-104246719-null-null.pc_agg_new_rank&utm_term=ubuntu怎么粘贴外部文本&spm=1000.2123.3001.4430) 

[5] [wsl中使用jupyter notebook_u010020404的博客-CSDN博客](https://blog.csdn.net/u010020404/article/details/121698485) 

[6] [搭建 Python 轻量级编写环境（WSL2+Jupyter 自动开启本地浏览器） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/158824489) 

[7] [Linux下配置安装JupyterNotebook，windows下通过浏览器直接连接使用_Together_CZ的博客-CSDN博客_linux安装jupyter notebook](https://blog.csdn.net/Together_CZ/article/details/105681245) 

[8] [Jupyter Notebook介绍、安装及使用教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/33105153) 
