# git工具笔记

### 1.git工具简介

   Git是目前世界上最先进的分布式版本控制系统。  Git是分布式版本控制系统，那么它就没有中央服务器的，每个人的电脑就是一个完整的版本库，这样，工作的时候就不需要联网了，因为版本都是在自己的电脑上。  适合分布式开发，强调个体。公共服务器压力和数据量都不会太大。速度快、灵活。任意两个开发者之间可以很容易的解决冲突。 

### 2.git的下载与使用

  通过git官网下载windows所使用的版本：

![1657810160712](D:\Program Files\Git\gitreo\summerHW\image\1657810160712.png)

  通过Git Bash调出控制台页面来进行进一步操作：

![1657809743370](D:\Program Files\Git\gitreo\summerHW\image\1657809743370.png)

### 3.文件相关操作

  通过git init将文件变成git可以管理的仓库

  文件内容的修改，查询文件状态

![1657810136539](D:\Program Files\Git\gitreo\summerHW\image\1657810136539.png)

![1657810140275](D:\Program Files\Git\gitreo\summerHW\image\1657810140275.png)

  文件的提交，这里要分为两步：

![1657810317100](C:\Users\17799\AppData\Roaming\Typora\typora-user-images\1657810317100.png)

  

![1657810372505](D:\Program Files\Git\gitreo\summerHW\image\1657810372505.png)

### 4.版本相关操作

  通过git reflog可以查找出版本：

![1657810562965](D:\Program Files\Git\gitreo\summerHW\image\1657810562965.png)

  并且通过reset可以对版本进行穿梭：

![1657810609204](D:\Program Files\Git\gitreo\summerHW\image\1657810609204.png)

### 5.分支相关操作

   git的分支功能特别的强大，它不需要将所有数据进行复制，只要重新创建一个分支的指针指向你需要从哪里开始创建分支的提交对象(commit)，然后进行修改再提交，那么新分支的指针就会指向你最新提交的这个commit对象，而原来分支的指针则指向你原来开发的位置，当你在哪个分支开发，HEAD就指向那个分支的最新提交对象commit。 通过分支，我们可以方便地来分布式地来对项目进行更新和维护。

  创建分支：

![1657811867683](D:\Program Files\Git\gitreo\summerHW\image\1657811867683.png)

  切换分支：

![1657811929811](D:\Program Files\Git\gitreo\summerHW\image\1657811929811.png)

  展示所有的分支：

![1657811884130](D:\Program Files\Git\gitreo\summerHW\image\1657811884130.png)

  

  对分支进行合并：

![1657812867187](D:\Program Files\Git\gitreo\summerHW\image\1657812867187.png)

  当合并的两个文件在同一个位置有不同的语句时，需要手动对合并进行操作。

### 6.push与pull操作

  用http方式时，需要输入账号密码进行连接，连接成功后：

![1657813394760](D:\Program Files\Git\gitreo\summerHW\image\1657813394760.png)

  申请SSH密钥，并且在github上进行加载

![1657813425441](D:\Program Files\Git\gitreo\summerHW\image\1657813425441.png)

![1657813429861](D:\Program Files\Git\gitreo\summerHW\image\1657813429861.png)

  在SSH的方式下，进行push和pull操作：

![1657813474410](D:\Program Files\Git\gitreo\summerHW\image\1657813474410.png)

### 7.问题及解决

  ①在上传材料至github后，发现md中的图片不会显示，如下所示：

<img src="C:\Users\17799\AppData\Roaming\Typora\typora-user-images\1657814887058.png" alt="1657814887058" style="zoom: 67%;" />

  考虑到是图片路径的问题，因此上传所需要的图片至github并且更新图片的路径

  ②尝试分别在在github上手动更新README文件，在本地文件夹中手动更新README文件，结果出现如下错误：

![1657815398287](D:\Program Files\Git\gitreo\summerHW\image\1657815398287.png)

  查阅相关资料发现， 出现错误的主要原因是github中的README.md文件不在本地代码目录中，因此要先进行pull操作，将文件pull下来进行合并

### 8.参考资料

[1] [Git使用教程,最详细，最傻瓜，最浅显，真正手把手教 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/30044692) 

[2] [收藏了！Git 核心操作图解 (qq.com)](https://mp.weixin.qq.com/s/Fg5rht0k583YvHD0pMJ_BQ) 

[3] [GitHub仓库快速导入Gitee及同步更新 - Gitee.com](https://gitee.com/help/articles/4284#article-header1) 

[4] [尚硅谷Git入门到精通全套教程（涵盖GitHub\Gitee码云\GitLab）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1vy4y1s7k6?p=16&vd_source=2e2e5482b016884769e4190397c8bfbb) 