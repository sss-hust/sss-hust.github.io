---
title: Anaconda
date: 2025-04-16 11:57:26
categories:
- 编程与工具
tags:
- Anaconda
- conda环境管理
- Python
- Linux安装
---
## linux下安装

```
# 获取安装包
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
# 运行安装脚本
bash Anaconda3-2025.06-1-Linux-x86_64.sh
# 一路回车、yes即可安装成功
# 在.bashrc中添加环境依赖
# 添加Anaconda到系统环境变量
export PATH="/root/anaconda3/bin:$PATH"
source ~/.bashrc
```

## anaconda使用

在命令行可以直接使用，前面会显示（base等），如果没有显示，就需要使用

> conda init powshell

重新激活，否则切换环境后，powshell处不会更改

![1744775899572](1744775899572.png)

几个常用的指令

- conda info 查看当前环境信息
- conda info -e 查看所有环境信息
- conda activate name 激活name环境
- conda create -n xxx python=x.x 新建环境
- conda remove -n xxx --all 移除环境

