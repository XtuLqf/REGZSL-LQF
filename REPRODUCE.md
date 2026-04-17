# RE-GZSL 复现指南

## 1. 服务器环境

本文档按以下服务器环境编写：

- GPU: RTX 5090
- Driver: 580.126.09
- System CUDA: 13.0

作者在 [README.md](README.md) 中给出的原始环境是 Python 3.7、Torch 1.8.0+cu101、Torchvision 0.9.0。考虑到当前服务器已验证可正常运行 PyTorch cu130 版本，以下步骤直接使用适配 5090 的环境配置。

## 2. 创建虚拟环境

推荐使用 conda 创建独立环境，环境名为 `regzsl`。Python 建议使用 3.10，以兼顾当前 PyTorch 版本与常见科学计算包兼容性。

```bash
conda create -n regzsl python=3.10 -y
conda activate regzsl
```

## 3. 安装依赖

先安装 PyTorch cu130 版本：

```bash
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

再安装其余 Python 依赖：

```bash
pip install numpy scipy scikit-learn matplotlib pillow
```

可选自检：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4. 数据集准备

默认数据路径为 `--dataroot /home/st/pytorch/lqf/Dataset`，可通过命令行参数修改。数据集目录结构如下：

```text
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

若使用微调模式（`--fine_tuning`），还需要在项目根目录下准备以下目录和文件。当前仓库中 `fine_tuning_data/` 尚不存在，只有启用该模式时才需要自行创建。

```text
fine_tuning_data/{dataset}/
├── train_feature.npy
├── test_seen_feature.npy
├── test_unseen_feature.npy
├── train_label.npy
├── test_seen_label.npy
└── test_unseen_label.npy
```

## 5. 创建必要目录

训练脚本会向 `log/` 写入日志，但不会自动创建该目录。当前仓库中 `log/` 已存在；如果你在新环境或新拷贝中运行，请先检查，不存在时再创建：

```bash
test -d log || mkdir log
```

当前仓库中 `output/` 尚不存在，但以下目录会在训练过程中自动创建，无需手动操作：

- `output/RE-GZSL/` — 主模型权重保存
- `output/baseline/` — baseline 模型权重保存

## 6. 训练

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
python main.py --dataset CUB --dataroot /home/st/pytorch/lqf/Dataset --cub_att sent --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

> **注意（CUB）**：默认使用 `--cub_att sent`，即 1024D sentence embedding；如需属性版本，请显式传入 `--cub_att att`，对应 312D attributes。

**SUN**：
```bash
python main.py --dataset SUN --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

**FLO**：
```bash
python main.py --dataset FLO --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100 --mad 0.8 --seen_Neighbours 4 --gamma 0.8 --contrast_ratio 0.1 --manualSeed 438
```

> **注意**：以上命令中的超参数仅供参考，不同数据集可能需要不同的超参数配置以达到最优效果。请根据 `option.py` 中的参数说明进行调整，并通过 `--dataroot` 指定数据集路径。

### Baseline 训练（baseline.py）

```bash
python baseline.py --dataset AWA1 --way 8 --shot 32 --c_batch_size 5000 --syn_num 5000 --c_epoch 30 --epoch 100
```

## 7. 评估

使用 `eval.py` 加载已保存的模型进行评估：

```bash
python eval.py --dataset AWA2 --dataroot /home/st/pytorch/lqf/Dataset
```

默认评估 `RE-GZSL` 主模型；如需评估 `baseline` 权重，请显式添加 `--eval_baseline`：

```bash
python eval.py --dataset AWA2 --dataroot /home/st/pytorch/lqf/Dataset --eval_baseline
```

## 8. 输出说明

| 输出 | 路径 | 说明 |
|---|---|---|
| 最佳模型权重 | `output/RE-GZSL/best_model.pth` | 主模型训练保存 |
| Baseline 权重 | `output/baseline/best_model.pth` | Baseline 训练保存 |
| 训练日志 | `log/log_{dataset}_{time}.txt` | 每轮训练指标记录 |
| t-SNE 可视化 | `tsne.png` | 需在代码中开启 `tsne_flag=True` |

## 9. 完整参数列表

运行以下命令查看所有可用参数：

```bash
python main.py --help
```
