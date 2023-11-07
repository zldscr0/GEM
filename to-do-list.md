### 主要代码

##### 尽量不要修改

`run_trainer.py`

`core/trainer.py`

##### 复现代码

模型定义`core/model/replay/gem.py`

模型参数`config/gem.yaml`

##### 目前存在问题

- [x] 1.`gem.py`的模型还没有定义分类头

  已解决

- [x] 2.task0跑通，task1未跑通

  已全部跑通

- [ ] 3.memory和buffer应该是一类东西，需要再确定一下数据存储策略是什么

- [x] 4.现在做的似乎是任务级增量学习，而不是类增量学习

  已修改

  memory和buffer应该是一类东西，需要再确定一下数据存储策略是什么

- [x] 5.在旧任务上表现非常差（acc=0），需要查看一下输出策略是不是有问题（offset1

  已修改，确实是forward中offset1设置的问题

- [x] 6.每次输出的loss为0，需要检查observe函数输出的loss值是不是有问题

  已修改，`core/trainer`里的_train没有回传loss值

- [ ] 7.task0有312个batch，之后就只有82个batch
- [ ] 8.虽然旧任务acc不全为0，但是性能依旧差，怀疑是memory存储有问题

> 注：torch版本较低时会报一个错，建议安装较高版本的torch



##### 待做

~~1.跑通代码初步测试精度~~

2.给代码加上注释，详细排查复现后的问题

3.与原文精度对比