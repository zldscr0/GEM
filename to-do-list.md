### 主要代码

##### 尽量不要修改

`run_trainer.py`

`core/trainer.py`

改动：

1.`core/trainer.py`中的`_train`

##### 复现代码

模型定义`core/model/replay/gem.py`

模型参数`config/gem.yaml`

##### 目前存在问题

- [x] 1.`gem.py`的模型还没有定义分类头

  已解决

- [x] 2.task0跑通，task1未跑通

  已全部跑通

- [ ] 3.memory和buffer应该是一类东西，需要再确定一下数据存储策略是什么

  采用ring buffer的存储方式，类似队列

- [x] 4.现在做的似乎是任务级增量学习，而不是类增量学习

  已修改

- [x] 5.在旧任务上表现非常差（acc=0），需要查看一下输出策略是不是有问题（offset1

  已修改，确实是forward中offset1设置的问题

- [x] 6.每次输出的loss为0，需要检查observe函数输出的loss值是不是有问题

  已修改，`core/trainer`里的_train没有回传loss值

- [ ] 7.虽然旧任务acc不全为0，但是性能依旧差，怀疑是memory存储有问题

  (1把buffer删了，正在做对比实验

  (2把buffer删了后前面任务acc为0

  (3修改了ptloss的计算方法（不加掩码），删除buffer前面任务acc不为0

- [ ] 8.task0（20类）的性能按理来说不应该与其它版本的gem复现相差太大，如pyCIL中的gem实现，但task0的准确率稳定在0.63这个数字，跑了pyCIL库，准确率大概在0.8左右(同一个任务，超参数设置差不多)

  现在最高到了0.69，离图上的0.8还有一定距离，换了resnet(32)，新增cifar_resnet，仍然没有明显的提升（从resnet18换到resnet34也是这样）

  

- [ ] 9.pyCIL中的实现有mask，跟我的理解有一点出入，问题【4】与本问题有关系

  

> 注：torch版本较低时会报一个错，建议安装较高版本的torch



##### 待做

~~1.跑通代码初步测试精度~~

2.给代码加上注释，详细排查复现后的问题

3.与原文精度对比



#### 实验记录（部分）

##### 精度

| id   | task-0 | task-1 | task-2 | task-3 | task-4 | 训练时长(s)        |
| ---- | ------ | ------ | ------ | ------ | ------ | ------------------ |
| 1    | 0.64   | 0.385  | 0.257  | 0.190  | 0.156  | 5640.173743247986  |
| 2    | 0.63   | 0.395  | 0.280  | 0.225  | 0.192  | 7065.652120113373  |
| 3    | 0.63   | 0.42   | 0.310  | 0.260  | 0.218  | 8139.007668972015  |
| 4    | 0.670  | 0.480  | 0.367  | 0.328  | 0.284  | 14511.146846532822 |
| 6    | 0.690  | 0.480  | 0.380  |        |        |                    |

随着实验次数增加，实验结果的提升如上表所示，每次实验除了backbone的变动与epoch的变动，还有优化器的一些参数调整之外，还修改了很多很多细节性的东西，每次变动的部分未做详细记录，但可根据日志文件的记录观察模型结构变动和超参数调整带来的性能变化。



日志存储位置

1:`log/GEM-resnet18-epoch25-23-11-07-18-14-34.log`

2.【未记录保存位置】

3.`log/GEM-resnet18-epoch20-23-11-08-11-55-38.log`

4.（相较于3，在优化器配置中添加了动量参数和权重衰减系数）

`log/GEM-resnet18-epoch50-23-11-08-12-35-08.log`

5.

5.`GEM-resnet18-epoch20-23-11-08-11-48-48.log`

| id   | t1   | t2   | t3    | t4    | t5    | t6    | t7    | t8    | t9    | t10   |                   |
| ---- | ---- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----------------- |
| 5    | 0.75 | 0.46 | 0.360 | 0.273 | 0.262 | 0.217 | 0.213 | 0.182 | 0.173 | 0.157 | 10246.50859093666 |
|      |      |      |       |       |       |       |       |       |       |       |                   |
|      |      |      |       |       |       |       |       |       |       |       |                   |



![image-20231108003258670](C:\Users\hanabi\AppData\Roaming\Typora\typora-user-images\image-20231108003258670.png)

![image-20231108003322183](C:\Users\hanabi\AppData\Roaming\Typora\typora-user-images\image-20231108003322183.png)









当前最优模型参数（20231108存

```
{'data_root': '/data/bzx_yjy/cifar100', 'image_size': 32, 'pin_memory': False, 'augment': True, 'workers': 8, 'device_ids': 0, 'n_gpu': 1, 'seed': 1993, 'deterministic': True, 'epoch': 80, 'batch_size': 128, 'val_per_epoch': 5, 'optimzer': {'name': 'SGD', 'kwargs': {'lr': 0.1}}, 'lr_scheduler': {'name': 'StepLR', 'kwargs': {'gamma': 0.1, 'step_size': 30}}, 'warmup': 3, 'includes': ['headers/data.yaml', 'headers/device.yaml', 'headers/model.yaml', 'headers/optimizer.yaml', 'backbones/resnet12.yaml'], 'save_path': './', 'init_cls_num': 20, 'inc_cls_num': 20, 'task_num': 5, 'optimizer': {'name': 'SGD', 'kwargs': {'lr': 0.1, 'momentum': 0.95, 'weight_decay': 0.0002}}, 'backbone': {'name': 'resnet32', 'kwargs': None}, 'buffer': {'name': 'LinearBuffer', 'kwargs': {'buffer_size': 0, 'batch_size': 32, 'strategy': 'random'}}, 'classifier': {'name': 'GEM', 'kwargs': {'num_class': 100, 'feat_dim': 64, 'n_memories': 5120, 'n_task': 5, 'memory_strength': 0}}, 'rank': 0}
GEM(

momentum: 0.95
    nesterov: False
    weight_decay: 0.0002
```

