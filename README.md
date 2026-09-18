[原仓库](https://github.com/caiyuanhao1998/Retinexformer?tab=readme-ov-file)

# Retinexformer原论文

网络结构

<img src="img/README_img/pipeline.png" alt="pipeline" style="zoom: 80%;" />

原结果

<img src="img/README_img/seven_results.png" alt="seven_results" style="zoom:80%;" />



# 创建环境

原项目：PyTorch 1.11 + cu113，最多使用30系显卡

```shell
conda create -n Retinexformer python=3.7 -y
conda activate Retinexformer

conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch -y
# 这一步可能报错
# Retinexformer) root@C.32741530:/workspace$ python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)" Traceback (most recent call last): File "<string>", line 1, in <module> File "/venv/Retinexformer/lib/python3.7/site-packages/torch/__init__.py", line 199, in <module> from torch._C import * # noqa: F403 ImportError: /venv/Retinexformer/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
# 如果报错则安装 conda install "mkl=2024.0" -y
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop
# 如果lmdb报错，是版本问题
python -c "import lmdb; print(lmdb.__version__)"
pip uninstall -y lmdb
pip install --no-cache-dir "lmdb==1.4.1"
```

安装BasicSR

```shell
cd /workspace/Retinexformer
python setup.py develop --no_cuda_ext
```



40系列：cuda=11.8

```
git clone https://github.com/jaxhur/Retinexformer.git

conda create -n Retinexformer python=3.10 -y
conda activate Retinexformer

conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop


# 验证GPU
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"


cd ./Retinexformer
python setup.py develop --no_cuda_ext
```

```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

50系列：CUDA 13.0+

```
git clone https://github.com/jaxhur/Retinexformer.git

conda create -n retinex50 python=3.10 -y
conda activate retinex50

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop

# 验证GPU
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

cd ./Retinexformer
python setup.py develop --no_cuda_ext
```

```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```



# 数据集

```
apt install -y unzip
cd ./data
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2重命名
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"

unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```



# 预训练权重进行测试

- 下载在不同训练集上的[模型权重](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link)，放到`./pretrained_weights`

  ```
  # linux
  cd /workspace/Retinexformer
  # 下载
  gdown --folder "https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link" -O pretrained_weights
  # 查看
  ls pretrained_weights
  ```

- Self-ensemble策略：使得结果更好，只需要加上`--self_ensemble`

<img src="img/README_img/image-20260312235645304.png" alt="image-20260312235645304" style="zoom:80%;" />



## 注意：不使用GT

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



# LOLv1

训练：5080

- 耗时：4h30min
- batch_size = 8
- patch_size =128

```
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml
```

测试

- PSNR：23.887855
- SSIM：0.82384
- LPIPS：0.155264257887999
- 参数量(M)：1.605701
- FLOPS(G)：17.018388

```
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights experiments/RetinexFormer_LOL_v1/models/best_G.pth --dataset LOL-v1
```



# LOLv2-real

训练：5080

- batch_size = 8、patch_size =256会爆显存
- batch_size = 4、patch_size =256，耗时9h

```
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real.yml
```

测试

```
# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights experiments/RetinexFormer_LOL_v2_real/models/best_G.pth --dataset LOL-v2-real
```



# LOLv2-syn

训练

- 5080，耗时5小时左右

```
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml
```

测试

- PSNR：23.887855
- SSIM：0.82384
- LPIPS：0.155264257887999
- 参数量(M)：1.605701
- FLOPS(G)：17.018388

```
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights experiments/RetinexFormer_LOL_v2_synthetic/models/best_G.pth --dataset LOL-v2-syn
```

