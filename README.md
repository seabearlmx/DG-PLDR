## DG-PLDR: Official Project Webpage
This is an initial code (V1)

## Pytorch Implementation
### Installation
Clone this repository.
```
git clone https://github.com/seabearlmx/DG-PLDR.git
cd DG-PLDR
```
Install following packages.

Refer to [mmpretrain](https://github.com/open-mmlab/mmpretrain)

1. conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
2. conda activate open-mmlab
3. pip install openmim
4. git clone https://github.com/open-mmlab/mmpretrain.git
5. cd mmpretrain
6. mim install -e .
7. mim install mmengine
8. copy /DG-PLDR/configs/ into /mmpretrain/configs/
9. copy /DG-PLDR/mmpretrain/datasets/ into /mmpretrain/mmpretrain/datasets/


### How to Run DG-PLDR
1. Download datasets of each benchmark. Please send email to seabearlmx@gmail.com to obtain datasets. Note that please use organization email.

2. You should modify the path in **"<path_to_dg-pldr>/configs/_base_/dataset/*.py"** according to your dataset path.

3. You should download pre-trained models on [mmpretrain](https://github.com/open-mmlab/mmpretrain) and modify the path in **"<path_to_dg-pldr>/configs/"** (e.g., in configs/apple/) according to your models path.

4. You can train DG-PLDR with following commands.
```
<path_to_dg-pldr>$ CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh ./configs/apple/eva02/eva02_dg_apple.py  2   # Train DG-Apple benchmark using eva02 models
<path_to_dg-pldr>$ CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh ./configs/apple/eva02/eva02_src_apple.py  2   # Train Src-Apple benchmark using eva02 models
<path_to_dg-pldr>$ CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh ./configs/apple/eva02/eva02_trg_apple.py  2   # Train Trg-Apple benchmark using eva02 models
```

5. You can test DG-PLDR with following commands.
```
<path_to_dg-pldr>$ CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_test.sh ./configs/apple/eva02/eva02_dg_apple.py ./work_dirs/xxx.pth  2
```

6. You can modify "frozen_stages" in the config files to adjust the fine-tuning strategy.

## Acknowledgments
Our pytorch implementation is heavily derived from [mmpretrain](https://github.com/open-mmlab/mmpretrain).
Thanks to the mmpretrain implementations.
