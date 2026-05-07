---
title: git使用
date: 2025-08-05 11:27:34
categories:
- 编程与工具
tags:
- Git
- SSH密钥
- rebase
- 版本控制
- 分支管理
---
首先配置ssh连接远程仓库，方便后续拉取代码

```
# 生成密钥，采用rsa算法，最后的-C表示备注，可选
ssh-keygen -t rsa -C "xxx@xxx.com"
# 复制公钥（这里文件名可以在生成的时候修改，以防冲突）
cat ~/.ssh/id_rsa.pub
```

将公钥复制到github或者其它代码库里，即可完成配置

### git指令(参考[教程](https://www.zhihu.com/question/594294987/answer/1962601516507575022))

#### 基本

- git add .
- git commit -m ""
- git push
- git pull    直接合并分支
- git fetch   只改变本地远程分支记录
- git clone
- git checkout -b fix/xxx
- git push origin fix/xxx

#### 提高

- git merge main  将main上的修改以及git记录都融入到当前分支，会出现很多跳跃关系，开发记录很复杂
- git rebash main  将当前分支开发记录接到main的最新节点上，一般只在自己的私人分支上使用，如果出现冲突，解决完后可以使用git rebase --continue继续，或者--abort放弃
- git commit --amend  修正上一次提交的信息，同时也可以将新文件加入上一次提交
- git rebase -i [基准commit]  整理commit信息，合并多个
- git reset --soft/mixed/hard  commit_id  三个参数依次为移动head指针，保存代码与暂存区重新commit；保存代码，清空暂存区；全部回退
- git reflog  最后的黑匣子
- git revert commit_id  把有问题的commit反向抵消，适用于代码已经合并到生产环境

#### 进阶

- git stash /git stash pop  将工作目录与暂存区所有打包起来，避免切换分支失去；pop为取出代码
- git cherry-pick commit_id  将目标提交复制到当前分支
- git bisect  二分查找出问题的提交

#### 实践

```
git rebase -i <commit_id>
git rebase --continue
git rebase --abort
```

交互式变基，允许穿越回过去的某个时间点，重新编排从那时起到现在的所有提交，可以把很多琐碎的提交合并，pick为保留，squash为合并到上一个提交；解决完冲突后，使用命令继续变基；一旦不想变基了，abort放弃所有回到开始变基前。
