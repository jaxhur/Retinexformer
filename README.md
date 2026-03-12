[原仓库](https://github.com/caiyuanhao1998/Retinexformer?tab=readme-ov-file)

# 实验结果

- Results on LOL-v1, LOL-v2-real, LOL-v2-synthetic, SID, SMID, SDSD-in, SDSD-out, and MIT Adobe FiveK datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1UCpHh3MkV4bxzWgiiULnb3BOPWS_8crP?usp=drive_link)

- Results on LOL-v1, LOL-v2-real, and LOL-v2-synthetic datasets with the same test setting as KinD, LLFlow, and recent diffusion models can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1Kbq9pASf1O_0Y9QMc88obQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1_ugNFblIYOCIam4cVJiXVX1aBGEYhn1o?usp=drive_link).

- Results on the NTIRE 2024 low-light enhancement dataset can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1bzICalpU1RprfnepLsMyCYT569UWTTlV?usp=sharing)

- Results on LIME, NPE, MEF, DICM, and VV datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1cqBwmuXk83h6u1NZJVbfkg?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1rWa_WRX5bqlW2HnBNMUGFKWrou7gIQpO?usp=drive_link)

- Results on ExDark nighttime object detection can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1ZvoPzYQePRc80-o7rrJuRQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1nZQnwKkGvswv--JunzgBLTXVRlOdcxb6?usp=sharing). Please use [this repo](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_high/IAT_mmdetection) to run experiments on the ExDark dataset

- Results of some compared baseline methods are shared on [Google Drive](https://drive.google.com/drive/folders/1P75bv6jBp8UxcLDhqMACvIQXck2sS9da?usp=sharing) and [Baidu Disk




# 复现

不复现NTIRE 2024

1、配置环境

```shell
conda create -n Retinexformer python=3.7
conda activate Retinexformer

conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

2、安装BasicSR

```shell
python setup.py develop --no_cuda_ext
```

3、下载数据

- [LOL-v1](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)
- [LOL-v2](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)
- [SID](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)
- [SMID](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG)
- [SDSD-indoor](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)
- [SDSD-outdoor](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)
- [MIT-Adobe FiveK](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing)
  - 原始数据集是RAW照片，而不是已经处理好的 JPG/PNG
  - 提供的链接中的图片已经处理好了
- 没有ground truth：
  - [LIME, NPE, MEF, DICM, VV](https://drive.google.com/drive/folders/1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp?usp=drive_link)
- 注意事项
  - SMID、SDSD-indoor、SDSD-outdoor是分卷压缩包，需要`.zip` 和 `.z01` 等放在同一个文件夹，解压`.zip`
  - 下载[text_list.txt](https://drive.google.com/file/d/199qrfizUeZfgq3qVjrM74mZ_nlacgwiP/view?usp=sharing)，放入`data/SMID/SMID_Long_np/`

Linux：

```
# 0) 在项目根目录
cd /workspace/Retinexformer

# 1) 安装下载/解压工具
pip install -U gdown
sudo apt-get update
sudo apt-get install -y unzip p7zip-full




# 2) 建目录
mkdir -p data/SMID data/SDSD data/LOLv1 data/LOLv2 data/SID data/FiveK
# 下载数据
# LOL-v1 (单文件)
gdown --fuzzy "https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing" -O data/LOLv1/LOLv1.zip

# LOL-v2 (单文件)
gdown --fuzzy "https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing" -O data/LOLv2/LOLv2.zip

# SID (文件夹)
gdown --folder "https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1" -O data/SID

# SMID (文件夹，含分卷压缩)
gdown --folder "https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG" -O data/SMID

# SDSD-indoor/outdoor (同一个文件夹，含分卷压缩)
gdown --folder "https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6" -O data/SDSD

# FiveK (单文件)
gdown --fuzzy "https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing" -O data/FiveK/FiveK.zip

# README要求的 text_list.txt
mkdir -p data/SMID/SMID_Long_np
gdown --fuzzy "https://drive.google.com/file/d/199qrfizUeZfgq3qVjrM74mZ_nlacgwiP/view?usp=sharing" -O data/SMID/SMID_Long_np/text_list.txt


# 分卷解压
cd data/SMID
unzip -o smid.zip
```

目录结构：


```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |--SDSD
    |    |    |--indoor_static_np
    |    |    |    |--input
    |    |    |    |    |--pair1
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--pair2
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |    |--GT
    |    |    |    |    |--pair1
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--pair2
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |--outdoor_static_np
    |    |    |    |--input
    |    |    |    |    |--MVI_0898
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--MVI_0918
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |    |--GT
    |    |    |    |    |--MVI_0898
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--MVI_0918
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |--SID
    |    |    |--short_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--long_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--SMID
    |    |    |--SMID_LQ_np
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--SMID_Long_np
    |    |    |    |--text_list.txt
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--FiveK
    |    |    |--train
    |    |    |    |--input
    |    |    |    |    |--a0099-kme_264.jpg
    |    |    |    |    |--a0101-kme_610.jpg
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--a0099-kme_264.jpg
    |    |    |    |    |--a0101-kme_610.jpg
    |    |    |    |     ...
    |    |    |--test
    |    |    |    |--input
    |    |    |    |    |--a4574-DSC_0038.jpg
    |    |    |    |    |--a4576-DSC_0217.jpg
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--a4574-DSC_0038.jpg
    |    |    |    |    |--a4576-DSC_0217.jpg
    |    |    |    |     ...
    |    |--NTIRE
    |    |    |--train
    |    |    |    |--input
    |    |    |    |    |--1.png
    |    |    |    |    |--3.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--1.png
    |    |    |    |    |--3.png
    |    |    |    |     ...
    |    |    |--minival
    |    |    |    |--input
    |    |    |    |    |--1.png
    |    |    |    |    |--31.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--1.png
    |    |    |    |    |--31.png
    |    |    |    |     ...

```


 

4、测试

- 下载在不同训练集上的[模型权重](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link)，放到`./pretrained_weights`

  ```
  # linux
  cd /workspace/Retinexformer
  pip install -U gdown
  mkdir -p pretrained_weights
  # 下载
  gdown --folder "https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link" -O pretrained_weights
  # 查看
  ls pretrained_weights
  
  ```

  

- Self-ensemble策略：使得结果更好，只需要加上`--self_ensemble`

测试命令

```shell
# activate the environment
conda activate Retinexformer

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# SDSD-indoor
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_indoor.yml --weights pretrained_weights/SDSD_indoor.pth --dataset SDSD_indoor

# SDSD-outdoor
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_outdoor.yml --weights pretrained_weights/SDSD_outdoor.pth --dataset SDSD_outdoor

# FiveK
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_FiveK.yml --weights pretrained_weights/FiveK.pth --dataset FiveK

# NTIRE
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_NTIRE.yml --weights pretrained_weights/NTIRE.pth --dataset NTIRE --self_ensemble

# MST_Plus_Plus trained with 4 GPUs on NTIRE 
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_4x1800.yml --weights pretrained_weights/MST_Plus_Plus_4x1800.pth --dataset NTIRE --self_ensemble

# MST_Plus_Plus trained with 8 GPUs on NTIRE 
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_8x1150.yml --weights pretrained_weights/MST_Plus_Plus_8x1150.pth --dataset NTIRE --self_ensemble

```

LLFlow、KinD、最近一些 diffusion 模型 相同的测试设置：使用ground truth的均值增强模型输出``--GT_mean` `

- 不推荐使用，不够公平、真实应用场景的测试通常拿不到 ground truth

```shell
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1 --GT_mean

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real --GT_mean

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic --GT_mean
```

`Enhancement/utils.py` 提供了`my_summary()` 用于**统计模型的参数量（Params）和计算复杂度（FLOPs）**

```shell
from utils import my_summary
my_summary(RetinexFormer(), 256, 256, 3, 1)
```


&nbsp;

5、训练

- 训练：[训练日志](https://drive.google.com/drive/folders/1HU_wEn_95Hakxi_ze-pS6Htikmml5MTA?usp=sharing)

```shell
# activate the enviroment
conda activate Retinexformer

# LOL-v1
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml

# SID
python3 basicsr/train.py --opt Options/RetinexFormer_SID.yml

# SMID
python3 basicsr/train.py --opt Options/RetinexFormer_SMID.yml

# SDSD-indoor
python3 basicsr/train.py --opt Options/RetinexFormer_SDSD_indoor.yml

# SDSD-outdoor
python3 basicsr/train.py --opt Options/RetinexFormer_SDSD_outdoor.yml

# FiveK
python3 basicsr/train.py --opt Options/RetinexFormer_FiveK.yml
```






