## 棋类训练程序

### 1.Description

使用自博弈保存的数据进行神经网络训练

### 2.Env

- Python3
- Pytorch
- einops

### 3.Directory

```txt
│  main.py                  # 主函数
│  play_main.py             # 对弈主函数
│  to_jit.py                # 转换为jit模型
│  train_net.py             # 训练模型脚本
│
├─backbone                  # 神经网络backbone
│
├─datasets                  # 数据集处理
│  │  conn6_dataset.py      # 自定义dataset类
│  │  data_type.py
│  │  README.md
│
├─module                    # 各种深度学习模块
│  │  test_lr.ipynb
│  │
│  ├─lr_scheduler           # 学习率
│  │  │  cosine.py
│  │  │  exponential.py
│  │  │  linear.py
│  │  │  noam.py
│  │  │  no_decay.py
│  │  │  warmup.py
│  │
│  └─optim                  # 优化器
└─utils                     # 一些常用的功能
    │  average_meter.py     # 平均累计
    │  bar.py               # 进度条
    │  file_buffer.py       # 文件缓存
    │  helper.py            # 常用函数
    │  log.py               # 日志类
    │  write_mixin.py

```

### 4.Running


