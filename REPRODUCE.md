# RE-GZSL 复现指南

## 1. 依赖安装

```bash
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy scipy scikit-learn matplotlib pillow
```

## 2. 数据集准备

默认数据路径为 `--dataroot /data1/wuyao/dataset/gzsl_data`，可通过命令行参数修改。数据集目录结构如下：

```
{dataroot}/
├── AWA1/
│   ├── res101.mat
│   └── att_splits.mat
├── AWA2/
│   ├── res101.mat
│   └── att_splits.mat
├── CUB/
│   ├── res101.mat
│   ├── att_splits.mat
│   └── sent_splits.mat      # 可选，使用 --cub_att sent 时需要
├── SUN/
│   ├── res101.mat
│   └── att_splits.mat
└── FLO/
    ├── data.mat
    └── label.mat
```

若使用微调模式（`--fine_tuning`），还需要在项目根目录下准备：

```
fine_tuning_data/{dataset}/
├── train_feature.npy
├── test_seen_feature.npy
├── test_unseen_feature.npy
├── train_label.npy
├── test_seen_label.npy
└── test_unseen_label.npy
```

## 3. 创建必要目录

训练前需手动创建 `log/` 目录（代码中未自动创建，否则写日志时会报 `FileNotFoundError`）：

```bash
mkdir log
```

以下目录会在训练过程中自动创建，无需手动操作：
- `output/RE-GZSL/` — 主模型权重保存
- `output/baseline/` — baseline 模型权重保存

## 4. 训练

### 主模型训练（main.py）

**AWA1**：
```bash
python main.py --dataset AWA1 --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

**AWA2**：
```bash
python main.py --dataset AWA2 --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

**CUB**：
```bash
python main.py --dataset CUB --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

**SUN**：
```bash
python main.py --dataset SUN --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

**FLO**：
```bash
python main.py --dataset FLO --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

> **注意**：以上命令中的超参数仅供参考，不同数据集可能需要不同的超参数配置以达到最优效果。请根据 `option.py` 中的参数说明进行调整。通过 `--dataroot` 指定数据集路径。

### Baseline 训练（baseline.py）

```bash
python baseline.py --dataset AWA1 --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100
```

## 5. 评估

使用 `eval.py` 加载已保存的模型进行评估：

```bash
python eval.py
```

> 注意：`eval.py` 中部分参数（如模型路径、`baseline` 标志）为硬编码，运行前请根据需要修改源文件中的对应变量。

## 6. 输出说明

| 输出 | 路径 | 说明 |
|---|---|---|
| 最佳模型权重 | `output/RE-GZSL/best_model.pth` | 主模型训练保存 |
| Baseline 权重 | `output/baseline/best_model.pth` | Baseline 训练保存 |
| 训练日志 | `log/log_{dataset}_{time}.txt` | 每轮训练指标记录 |
| t-SNE 可视化 | `tsne.png` | 需在代码中开启 `tsne_flag=True` |

## 7. 完整参数列表

运行以下命令查看所有可用参数：

```bash
python main.py --help
```
