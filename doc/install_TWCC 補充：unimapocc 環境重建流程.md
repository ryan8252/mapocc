## Environment Setup
### step 1. Install environment
Install environment
```
# 1. 建立基礎環境 (維持 Python 3.8 確保與舊套件的最大相容性)
/home/u2336262/miniconda3/bin/conda create --prefix /home/u2336262/.conda/envs/ProtoOcc_final python=3.8 -y
source /home/u2336262/miniconda3/etc/profile.d/conda.sh
conda activate /home/u2336262/.conda/envs/ProtoOcc

# 2. 清空並載入國網的 CUDA 11.7 模組
module purge
module load cuda/11.7
module load openmpi/4.1.6_ucx1.14.1_cuda12.3

# 3. 設定環境變數 (讓系統知道 CUDA 的家在哪)
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. 安裝 PyTorch 2.0.1 
# (強烈建議鎖定 2.0.1，如果安裝最新版 PyTorch 2.x 會導致 mmcv-full 無法編譯)
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117

# 5. 安裝必要套件
pip install -U openmim
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

pip install ipython
pip install torchmetrics==0.11.4
```

#### After setting up the environment or pulling the Docker image, run `git clone` and `pip install`.
```
cd ProtoOcc

cd ./mmdetection3d
pip install -v -e .

cd ../projects
pip install -v -e .
cd ..
```

### Step 2. Download and unzip the [nuScenes dataset](https://www.nuscenes.org/download) (including panoptic files), and get the gts folder from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction).
For auxiliary task (perspective semantic segmentation), replace `v1.0-trainval/category.json` within the received folder with the `category.json` in `./data/nuscenes/v1.0-trainval`, and move `v1.0-trainval/panoptic.json` from the received folder to `./data/nuscenes/v1.0-trainval`.

### step 3. Prepare nuScenes dataset as below:
```shell script
└── ProtoOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval
                ├── panoptic.json
                ├── category.json # (from nuscenes panoptic)
                ├── ...
            ├── sweeps 
            ├── samples
            ├── panoptic
            └── gts 
```


### step 4. Preprocess for training

Create the pkl or download [Here](https://drive.google.com/drive/folders/1aiG4wmsj4Q7cJBQ4-H1lrG8wiFIhoa-W?usp=drive_link):
```shell script
python tools/create_data_bevdet.py
```

Run code below for `pc_panoptic`
```shell script
python tools/data_converter/prepare_panoptic.py
```


### step 5. Download [ckpts](https://drive.google.com/drive/folders/1e459AGnjwtatnakv2kyOR2beweJTk03e?usp=sharing) to `ProtoOcc/ckpts/`:

### The final directory should be organized as follows 
```shell script
└── ProtoOcc/
    ├── data
        └── nuscenes
            ├── v1.0-trainval 
            ├── sweeps  
            ├── samples
            ├── panoptic
            ├── pc_panoptic
            ├── gts 
            ├── bevdetv2-nuscenes_infos_train.pkl 
            └── bevdetv2-nuscenes_infos_val.pkl
    ├── ckpts
        ├── bevdet-r50-4d-depth-cbgs_depthnet_modify.pth # (Renamed 'depth_net' in 'state_dict' for pretrained weights)
        ├── bevdet-r50-4dlongterm-stereo-cbgs.pth
    ├── doc
    ├── mmdetection3d 
    ├── projects
    ├── requirements
    ├── tools
    ├── plot
    └── README.md
```

---
## Run TWCC GPUs
1. 需要先寫一個腳本 (bash script), 格式請見 `bash_script_example.sh`
2. 派送工作
    ```
    sbatch <your-bash-script>.sh
    ```
3. 派送成功後，會看到: 
    ```
    "Submitted batch job <your JOBID>"
    ```

## Check TWCC GPUs 
1. 查看教授帳號下所有正在跑的 process
    ```
    squeue -u u2336262
    ```
2. 查看特定 process 之 GPU uasge
    ```
    srun --jobid <your JOBID> nvidia-smi
    ```

---
## 2026-04-13 補充：`unimapocc` 環境重建流程（同一台機器、同一個 Linux 帳號）

如果同一個 Linux 帳號底下有多個專案或多個 conda env，`pip` 可能會因為使用者層級設定而把套件裝到 `~/.local/lib/python3.8/site-packages`，進而造成：

- `pip install` 顯示成功，但 `python import torch` 失敗
- `pip install -e .` 的 `egg-link` 指到別人的 repo 路徑
- `torch` 實際吃到 CUDA 12.1 版，但 TWCC module 載的是 CUDA 11.7，導致 extension 編譯失敗

以下流程是目前 `unimapocc` 在 TWCC 上較安全的安裝方式。

### Step A. 建立並啟用 `unimapocc`
```bash
/home/u2336262/miniconda3/bin/conda create --prefix /home/u2336262/.conda/envs/unimapocc python=3.8 -y
source /home/u2336262/miniconda3/etc/profile.d/conda.sh
conda activate /home/u2336262/.conda/envs/unimapocc
```

### Step B. 載入 TWCC CUDA 模組
```bash
module purge
module load cuda/11.7
module load openmpi/4.1.6_ucx1.14.1_cuda12.3

export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Step C. 隔離 `~/.local` 與其他專案的 `PYTHONPATH`
```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER
```

### Step D. 在 `unimapocc` env 內覆蓋 user-level pip 設定
```bash
cat > /home/u2336262/.conda/envs/unimapocc/pip.conf <<'EOF'
[install]
user = false
EOF
```

### Step E. 安裝 / 重裝核心套件
先更新安裝工具：
```bash
python -m pip install --no-user --force-reinstall pip setuptools wheel
```

強制安裝 CUDA 11.7 對應的 PyTorch wheel：
```bash
python -m pip uninstall -y torch torchvision torchaudio triton
python -m pip install --no-user --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2+cu117
```

安裝 OpenMMLab 與 ProtoOcc 需要的套件：
```bash
python -m pip install --no-user --force-reinstall -U openmim
python -m pip install --no-user --force-reinstall \
  mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
python -m pip install --no-user --force-reinstall mmdet==2.28.2
python -m pip install --no-user --force-reinstall mmsegmentation==0.30.0

python -m pip install --no-user --force-reinstall \
  ipython \
  torchmetrics==0.11.4 \
  lyft_dataset_sdk \
  networkx==2.2 \
  numba==0.53.0 \
  numpy \
  nuscenes-devkit \
  plyfile \
  scikit-image \
  tensorboard \
  trimesh==2.35.39
```

### Step F. 安裝 repo 內的 editable packages
請一律使用 `python -m pip`，不要直接打 `pip`。
```bash
cd /home/u2336262/Desktop/artc_2026/unimapocc/ProtoOcc/mmdetection3d
python -m pip install --no-user -e .

cd /home/u2336262/Desktop/artc_2026/unimapocc/ProtoOcc/projects
python -m pip install --no-user -e .
```

### Step G. 驗證目前真的吃到 `unimapocc` env
```bash
python -m pip --version
python -c "import pip; print(pip.__file__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
python -c "import mmcv, mmdet, mmseg; print(mmcv.__version__, mmdet.__version__, mmseg.__version__)"
```

如果最後一行在 import `mmcv` 時出現：

```bash
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory
```

表示 `mmcv` 載入 `cv2` 時吃到需要 GUI/X11 library 的 OpenCV wheel，但目前 CUDA container 沒有 `libxcb`。在 headless 訓練環境中不要補系統 X11 library，直接把 OpenCV 換成 headless wheel：

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python -m pip install --no-user --force-reinstall --no-deps opencv-python-headless==4.8.1.78
python -m pip install --no-user --force-reinstall numpy==1.23.5 networkx==2.2

python -c "import numpy as np; print(np.__version__)"
python -c "import networkx as nx; print(nx.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import mmcv, mmdet, mmseg; print(mmcv.__version__, mmdet.__version__, mmseg.__version__)"
```

理想情況：

- `pip.__file__` 在 `/home/u2336262/.conda/envs/unimapocc/...`
- `torch.version.cuda` 顯示 `11.7`
- 不應該再看到 `~/.local/lib/python3.8/site-packages/pip`

### Step H. 若 `projects` 的 editable install 指到舊路徑
如果 `ProtoCcc_plugin.egg-link` 還殘留在 `~/.local` 並指向舊 repo，可清掉後重裝：
```bash
rm -f /home/u2336262/.local/lib/python3.8/site-packages/ProtoCcc-plugin.egg-link
rm -rf /home/u2336262/.local/lib/python3.8/site-packages/ProtoCcc_plugin.egg-info

cd /home/u2336262/Desktop/artc_2026/unimapocc/ProtoOcc/projects
python -m pip install --no-user -e .
```

### Step I. 送訓練前的注意事項
1. `ProtoOcc/projects/configs/ProtoOcc/ProtoOcc_multitask_stage1.py` 目前會從：
   ```
   work_dirs/ProtoOcc_1key/ProtoOcc_1key.pth
   ```
   載入 pretrained checkpoint。送訓練前請先確認這個檔案存在。

2. 若是同一帳號多人共用機器，建議所有 sbatch 腳本都加上：
   ```bash
   export PYTHONNOUSERSITE=1
   unset PYTHONPATH
   unset PIP_USER
   ```

3. 本 repo 的 Stage 1 訓練腳本可直接參考 `TWCC/train_multitask_stage1_twcc.sh`
4. `bash_script_example.sh` 內的 `#SBATCH --account=your_project_account` 只是 placeholder，實際送 job 前一定要改成你有權限使用的 project id。
   以這次 `sbatch` 回傳的 wallet info 為例，應填：
   ```bash
   #SBATCH --account=MST113104
   ```

---
## 2026-06-03 補充：`25a-lgn01` 新主機上的 `unimapocc` 環境建立流程

這次是在新主機 `25a-lgn01` 上建立環境，帳號仍是 `u2336262`，但此主機沒有：

```bash
/home/u2336262/miniconda3
```

因此不能再使用舊流程中的：

```bash
/home/u2336262/miniconda3/bin/conda
source /home/u2336262/miniconda3/etc/profile.d/conda.sh
```

此主機目前可用 TWCC module 提供的 miniconda：

```bash
module load miniconda3/26.1.1
```

目前 repo 位置為：

```bash
/home/u2336262/Desktop/artc_2026/mapocc
```

其中已確認有：

```bash
/home/u2336262/Desktop/artc_2026/mapocc/mmdetection3d
/home/u2336262/Desktop/artc_2026/mapocc/projects
```

### 重要安全提醒

以下流程不會修改系統環境，也不會影響其他 Linux 帳號。`module load`、`module purge`、`export`、`unset` 都只影響目前這個 terminal/session。

但如果多人共用同一個 Linux 帳號 `u2336262`，則以下路徑屬於共用帳號底下的使用者空間，仍可能互相影響：

```bash
/home/u2336262/.conda/envs/unimapocc
/home/u2336262/.conda
/home/u2336262/.local
```

因此建立 env 前會先檢查 `/home/u2336262/.conda/envs/unimapocc` 是否已存在。若已存在，請先確認是否為自己的環境，不要直接覆蓋。

### Step A. 確認目前主機與專案位置

```bash
whoami
hostname
pwd

find /home/u2336262/Desktop/artc_2026/mapocc -maxdepth 3 -type d \( -name mmdetection3d -o -name projects -o -name ProtoOcc \)
```

預期目前資訊：

```bash
whoami    # u2336262
hostname  # 25a-lgn01
pwd       # /home/u2336262/Desktop/artc_2026/mapocc
```

### Step B. 載入 TWCC module 版 miniconda

這台主機沒有私人安裝的 `/home/u2336262/miniconda3`，因此改用 TWCC module 提供的 miniconda。

```bash
module purge
module load miniconda3/26.1.1

source "$(conda info --base)/etc/profile.d/conda.sh"
```

### Step C. 處理 Anaconda default channels 的 ToS 問題

新版 Anaconda conda 第一次使用 default channels 時，可能會出現：

```bash
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

`ToS` 是 `Terms of Service`，也就是服務條款。如果這個 Linux 帳號是多人共用，建議先避免用 Anaconda default channels，改用 `conda-forge` 建環境，避免替共用帳號接受 Anaconda channel 條款。

建議使用此方式建立環境：

```bash
if [ -e /home/u2336262/.conda/envs/unimapocc ]; then
  echo "ERROR: /home/u2336262/.conda/envs/unimapocc already exists."
  echo "Please check whether this env is yours before modifying it."
  exit 1
fi

conda create --prefix /home/u2336262/.conda/envs/unimapocc \
  --override-channels -c conda-forge \
  python=3.8 -y

conda activate /home/u2336262/.conda/envs/unimapocc
```

如果確認 `u2336262` 是自己可管理的帳號，且可以接受 Anaconda default channels 的 ToS，也可以先執行：

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

然後用原本的 default channels 建立：

```bash
conda create --prefix /home/u2336262/.conda/envs/unimapocc python=3.8 -y
conda activate /home/u2336262/.conda/envs/unimapocc
```

### Step D. 載入 CUDA 11.7 與 OpenMPI module

`module load` 只影響目前 shell，不會修改系統設定或其他人的 terminal。

實測 `25a-lgn01` 目前沒有舊流程使用的：

```bash
cuda/11.7
openmpi/4.1.6_ucx1.14.1_cuda12.3
```

目前 `25a-lgn01` 實測可看到的 CUDA module 為：

```bash
cuda/12.6
cuda/13.0
```

且目前 `module avail openmpi` 沒有列出可用 OpenMPI module；`module spider cuda/11.7` 與 `module spider cuda/11.8` 都回報找不到。

因此下面這段是舊主機或有 CUDA 11.7 module 時才適用。若在 `25a-lgn01` 執行 `module load cuda/11.7` 出現 `module(s) are unknown`，不要直接改成 `cuda/12.6` 後繼續安裝舊版 `torch==2.0.1+cu117` / `mmcv-full cu117` 流程，因為後續 `mmdetection3d` 等 CUDA extension 編譯時可能會遇到 PyTorch CUDA 版本與系統 `nvcc` 版本不一致。

在 `25a-lgn01` 上較安全的方向：

1. 詢問 TWCC 是否有可用的 CUDA 11.7/11.8 module 或舊版環境 stack。
2. 使用 Singularity/Apptainer 容器，讓容器內提供 CUDA 11.7/11.8 userland，再用主機 driver 跑 GPU。
3. 若必須使用 `cuda/12.6`，則需要整組改 PyTorch / mmcv / mmdet3d 相容版本，不建議直接沿用此份 `cu117` 流程。

```bash
module load cuda/11.7
module load openmpi/4.1.6_ucx1.14.1_cuda12.3

export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### Step D alternative. `25a-lgn01` 使用 Singularity CUDA 11.7 container

`25a-lgn01` 已確認可使用：

```bash
apptainer version 1.4.3-1.el9
singularity-ce version 4.3.7
```

且下列測試已成功，表示 container 可以透過 `--nv` 使用主機 GPU：

```bash
module load singularity/4.3.7

singularity exec --nv docker://nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 nvidia-smi
```

`nvidia-smi` 裡顯示的 `CUDA Version: 13.0` 是主機 NVIDIA driver 支援上限，不代表 container 內 CUDA 變成 13.0。container 內 `nvcc` 已確認為 CUDA 11.7：

```bash
singularity exec --nv docker://nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 \
  bash -c 'command -v nvcc; nvcc --version; ls -l /usr/local/cuda'
```

預期會看到：

```bash
/usr/local/cuda/bin/nvcc
Cuda compilation tools, release 11.7, V11.7.99
```

建議先把 Docker image 固定成 `.sif`，避免每次執行都重新轉換：

```bash
module load singularity/4.3.7

mkdir -p /home/u2336262/Desktop/artc_2026/containers

singularity pull \
  /home/u2336262/Desktop/artc_2026/containers/cuda117-cudnn8-devel-ubuntu20.04.sif \
  docker://nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
```

之後進入 container shell。這裡不使用 `bash -lc`，避免 TWCC/Lmod 的 login-shell 初始化污染 container：

```bash
export SIF=/home/u2336262/Desktop/artc_2026/containers/cuda117-cudnn8-devel-ubuntu20.04.sif

singularity shell --cleanenv --nv \
  --bind /home/u2336262:/home/u2336262 \
  "$SIF"
```

進入 container 後，設定 CUDA 11.7 與 Step C 建好的 `unimapocc` conda env：

```bash
export ENV_PATH=/home/u2336262/.conda/envs/unimapocc

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$ENV_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

python -m pip --version
python -c "import sys; print(sys.executable)"
nvcc --version
```

如果 `python -c "import sys; print(sys.executable)"` 顯示：

```bash
/home/u2336262/.conda/envs/unimapocc/bin/python
```

就可以在這個 container shell 內繼續執行 Step F 到 Step K。不要在 host shell 裡安裝 `torch==2.0.1+cu117` / `mmcv-full cu117`。

### Step E. 隔離 `~/.local` 與其他專案的 `PYTHONPATH`

避免 pip 或 Python 吃到 `~/.local/lib/python3.8/site-packages` 裡其他專案殘留的套件。

```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER
```

建議之後所有 `sbatch` 腳本也加上這三行。

### Step F. 在 `unimapocc` env 內禁止 pip 裝到 user site

這只會寫入 `unimapocc` 這個 conda env，不會改全域 pip 設定。

```bash
cat > /home/u2336262/.conda/envs/unimapocc/pip.conf <<'EOF'
[install]
user = false
EOF
```

### Step G. 安裝 / 重裝基礎安裝工具

請一律使用 `python -m pip`，不要直接使用 `pip`。

```bash
python -m pip install --no-user --force-reinstall pip setuptools wheel
```

### Step H. 安裝 CUDA 11.7 對應的 PyTorch

此專案需要讓 PyTorch、CUDA module、mmcv-full 三者版本對齊。這裡固定使用 PyTorch `2.0.1+cu117`。

```bash
python -m pip uninstall -y torch torchvision torchaudio triton
python -m pip install --no-user --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2+cu117
```

若後續安裝 `projects` 時看到：

```bash
RuntimeError:
The detected CUDA version (11.7) mismatches the version that was used to compile
PyTorch (12.1).
```

表示目前 shell 用到的 `nvcc` 已經是 container 內的 CUDA 11.7，但 `unimapocc` env 裡的 PyTorch 仍是 `cu121` 版本。這不是 `ninja` 警告造成的；`ninja` 只影響編譯速度。

在 container shell 內先確認：

```bash
python -m pip --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
nvcc --version
```

若 `torch.version.cuda` 顯示 `12.1`，在同一個 container shell 內重裝 PyTorch `cu117`：

```bash
python -m pip uninstall -y torch torchvision torchaudio triton

python -m pip install --no-user --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2+cu117

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
```

確認 `torch.version.cuda` 是 `11.7` 後，再回去執行 `mmdetection3d` / `projects` 的 editable install。

#### H200 上 `torch==2.0.1+cu117` 會出現 `no kernel image`

在 Nano4 / `25a-hgpn*` H200 GPU 上，可能看到：

```bash
2.0.1+cu117 11.7 True NVIDIA H200
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

這代表 PyTorch 可以看到 GPU，但 `torch==2.0.1+cu117` wheel 不能在 H200 上執行 CUDA kernel。H200 是 compute capability `9.0`，因此 Nano4 H200 不建議走 `cu117`；保留 PyTorch `2.0.1` / OpenMMLab `1.x` 的最小改法是改成 CUDA 11.8 container + PyTorch `cu118` + mmcv-full `cu118`。

先建立 CUDA 11.8 SIF：

```bash
module load singularity/4.3.7

singularity pull \
  /home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif \
  docker://nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

進入 CUDA 11.8 container：

```bash
export SIF=/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif

singularity shell --cleanenv --nv \
  --bind /home/u2336262:/home/u2336262 \
  "$SIF"
```

在 container shell 內改裝 `cu118`：

```bash
export ENV_PATH=/home/u2336262/.conda/envs/unimapocc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$ENV_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER

python -m pip uninstall -y torch torchvision torchaudio triton mmcv-full mmcv

python -m pip install --no-user --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  torchaudio==2.0.2+cu118

python -m pip install --no-user --no-cache-dir --force-reinstall \
  mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
python -c "from mmcv.ops import get_compiler_version, get_compiling_cuda_version; print(get_compiler_version()); print(get_compiling_cuda_version())"
```

通過後，再重裝 `mmdetection3d` / `projects` editable packages，並重新跑 `TWCC_nano4/test_singularity_unimapocc_nano4.sh`。

重裝 PyTorch 時，pip 可能會順手把部分共享依賴升級，例如：

```bash
mmdet3d 1.0.0rc4 requires networkx<2.3,>=2.2, but you have networkx 3.1
```

這會影響 `mmdet3d` 相容性。先把 `networkx` 釘回 `2.2`，把 `numpy` 釘回 `1.23.5`，並補上 `ninja` 讓 CUDA extension 使用較快的 build backend。注意：後續安裝 `opencv-python-headless` 時要加 `--no-deps`，否則 pip 可能又會把 NumPy 拉回 `1.24+`，造成 `networkx==2.2` import 時出現 `np.int` 錯誤。

```bash
python -m pip install --no-user --force-reinstall networkx==2.2
python -m pip install --no-user --force-reinstall numpy==1.23.5
python -m pip install --no-user ninja

python -c "import networkx as nx; print(nx.__version__)"
python -c "import numpy as np; print(np.__version__)"
```

### Step I. 安裝 OpenMMLab 與專案依賴

```bash
python -m pip install --no-user --force-reinstall -U openmim

python -m pip install --no-user --force-reinstall \
  mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html

python -m pip install --no-user --force-reinstall mmdet==2.28.2
python -m pip install --no-user --force-reinstall mmsegmentation==0.30.0

python -m pip install --no-user --force-reinstall \
  ipython \
  torchmetrics==0.11.4 \
  lyft_dataset_sdk \
  networkx==2.2 \
  numba==0.53.0 \
  numpy==1.23.5 \
  nuscenes-devkit \
  plyfile \
  scikit-image \
  tensorboard \
  trimesh==2.35.39
```

### Step J. 安裝 repo 內 editable packages

這會在 `unimapocc` env 裡建立 editable link，指向目前新主機上的 `mapocc` repo。

```bash
cd /home/u2336262/Desktop/artc_2026/mapocc/mmdetection3d
python -m pip install --no-user -e .

cd /home/u2336262/Desktop/artc_2026/mapocc/projects
python -m pip install --no-user -e .
```

### Step K. 驗證目前真的吃到 `unimapocc` env

```bash
python -m pip --version
python -c "import pip; print(pip.__file__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.__file__)"
python -c "import mmcv, mmdet, mmseg; print(mmcv.__version__, mmdet.__version__, mmseg.__version__)"
```

理想情況：

- `pip.__file__` 在 `/home/u2336262/.conda/envs/unimapocc/...`
- `torch.version.cuda` 顯示 `11.7`
- `mmcv.__version__` 顯示 `1.7.2`
- `mmdet.__version__` 顯示 `2.28.2`
- `mmseg.__version__` 顯示 `0.30.0`
- 不應該看到 `~/.local/lib/python3.8/site-packages/pip`

### Step L. 暫時不要執行舊流程中的 `~/.local` 清除指令

舊補充流程中的以下指令是用來修復舊主機上已經污染的 editable install：

```bash
rm -f /home/u2336262/.local/lib/python3.8/site-packages/ProtoCcc-plugin.egg-link
rm -rf /home/u2336262/.local/lib/python3.8/site-packages/ProtoCcc_plugin.egg-info
```

在 `25a-lgn01` 新主機上，除非已確認 `projects` 的 editable install 指到舊 repo，否則不要一開始就執行這兩行。

---

## 2026-06-03 補充：`25a-lgn01` / H200 container 環境建立流程

### 使用限制

- 只在 `25a-lgn01` / Nano4 H200 上使用這段流程。
- miniconda module 只用來建立 `/home/u2336262/.conda/envs/unimapocc`。
- CUDA、`nvcc`、PyTorch、mmcv 編譯與訓練都在 Singularity container 內做。
- H200 不用 `cu117`，使用 `cu118`。
- H200 不用 OpenMMLab 預編譯 `mmcv-full` wheel，要從 source 編譯 `mmcv-full==1.7.2`。
- `mapocc/data` 指到 `/work/u2336262/data`，所以 container 和 sbatch 都要 bind `/work:/work`。

### Step A. 確認主機與 repo

```bash
whoami
hostname
cd /home/u2336262/Desktop/artc_2026/mapocc
pwd
```

預期：

```bash
u2336262
25a-lgn01
/home/u2336262/Desktop/artc_2026/mapocc
```

### Step B. 建立 `unimapocc` conda env

只在 env 不存在時執行。

```bash
module purge
module load miniconda3/26.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -e /home/u2336262/.conda/envs/unimapocc ]; then
  echo "ERROR: /home/u2336262/.conda/envs/unimapocc already exists."
  exit 1
fi

conda create --prefix /home/u2336262/.conda/envs/unimapocc \
  --override-channels -c conda-forge \
  python=3.8 -y
```

### Step C. 建立 CUDA 11.8 Singularity image

```bash
module purge
module load singularity/4.3.7

mkdir -p /home/u2336262/Desktop/artc_2026/containers

singularity pull \
  /home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif \
  docker://nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

### Step D. 進入 container

```bash
module purge
module load singularity/4.3.7

export SIF=/home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif

singularity shell --cleanenv --nv \
  --bind /home/u2336262:/home/u2336262 \
  --bind /work:/work \
  "$SIF"
```

### Step E. 在 container 內啟用 `unimapocc`

以下都在 `Singularity>` 裡執行。

```bash
export ENV_PATH=/home/u2336262/.conda/envs/unimapocc
export CUDA_HOME=/usr/local/cuda
export PATH=$ENV_PATH/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PIP_USER
```

```bash
python -m pip --version
python -c "import sys; print(sys.executable)"
nvcc --version
```

### Step F. 固定 pip 行為

```bash
cat > /home/u2336262/.conda/envs/unimapocc/pip.conf <<'EOF'
[install]
user = false
EOF

python -m pip install --no-user --force-reinstall pip setuptools wheel
```

### Step G. 安裝 PyTorch CUDA 11.8

```bash
python -m pip uninstall -y torch torchvision torchaudio triton

python -m pip install --no-user --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  torchaudio==2.0.2+cu118
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

### Step H. 從 source 編譯 `mmcv-full`

```bash
python -m pip uninstall -y mmcv-full mmcv
python -m pip install --no-user ninja

export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4

python -m pip install --no-user --no-cache-dir -v \
  --no-binary mmcv-full \
  mmcv-full==1.7.2
```

```bash
python -c "from mmcv.ops import get_compiler_version, get_compiling_cuda_version; print(get_compiler_version()); print(get_compiling_cuda_version())"
python -c "import torch; from mmdet.models.losses import FocalLoss; loss=FocalLoss(use_sigmoid=True).cuda(); pred=torch.randn(16,2,device='cuda'); target=torch.zeros(16,dtype=torch.long,device='cuda'); print(loss(pred,target))"
```

### Step I. 安裝 OpenMMLab 與專案依賴

```bash
python -m pip install --no-user --force-reinstall mmdet==2.28.2
python -m pip install --no-user --force-reinstall mmsegmentation==0.30.0

python -m pip install --no-user --force-reinstall \
  ipython \
  torchmetrics==0.11.4 \
  lyft_dataset_sdk \
  numba==0.53.0 \
  nuscenes-devkit \
  plyfile \
  scikit-image \
  tensorboard \
  trimesh==2.35.39

python -m pip install --no-user --force-reinstall numpy==1.23.5 networkx==2.2
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python -m pip install --no-user --force-reinstall --no-deps opencv-python-headless==4.8.1.78
python -m pip install --no-user ninja
python -m pip install --no-user --force-reinstall numpy==1.23.5 networkx==2.2
```

### Step J. 安裝 repo editable packages

```bash
cd /home/u2336262/Desktop/artc_2026/mapocc/mmdetection3d
MAX_JOBS=4 python -m pip install --no-user -e .

cd /home/u2336262/Desktop/artc_2026/mapocc/projects
MAX_JOBS=4 python -m pip install --no-user -e .
```

### Step K. 驗證環境

```bash
cd /home/u2336262/Desktop/artc_2026/mapocc

python -m pip --version
python -c "import pip; print(pip.__file__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
python -c "import numpy as np; print(np.__version__)"
python -c "import networkx as nx; print(nx.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import mmcv, mmdet, mmseg, mmdet3d; print(mmcv.__version__, mmdet.__version__, mmseg.__version__, mmdet3d.__version__)"
python -c "from mmcv.ops import get_compiler_version, get_compiling_cuda_version; print(get_compiler_version()); print(get_compiling_cuda_version())"
python -c "import torch; from mmdet.models.losses import FocalLoss; loss=FocalLoss(use_sigmoid=True).cuda(); pred=torch.randn(16,2,device='cuda'); target=torch.zeros(16,dtype=torch.long,device='cuda'); print(loss(pred,target))"
```

預期輸出重點：

```bash
pip ... from /home/u2336262/.conda/envs/unimapocc/lib/python3.8/site-packages/pip (python 3.8)
/home/u2336262/.conda/envs/unimapocc/lib/python3.8/site-packages/pip/__init__.py
2.0.1+cu118 11.8 NVIDIA H200
tensor([...], device='cuda:0')
1.23.5
2.2
4.8.1
1.7.2 2.28.2 0.30.0 1.0.0rc4
11.8
tensor(..., device='cuda:0')
```

### Step L. 檢查資料路徑

```bash
cd /home/u2336262/Desktop/artc_2026/mapocc

readlink -f data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
ls -lh data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
ls -lh data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
```

預期輸出重點：

```bash
/work/u2336262/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
... data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
... data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
```

container 內確認：

```bash
singularity exec --cleanenv --nv \
  --bind /home/u2336262:/home/u2336262 \
  --bind /work:/work \
  /home/u2336262/Desktop/artc_2026/containers/cuda118-cudnn8-devel-ubuntu20.04.sif \
  bash -c 'export ENV_PATH=/home/u2336262/.conda/envs/unimapocc; export PATH=$ENV_PATH/bin:/usr/local/cuda/bin:$PATH; cd /home/u2336262/Desktop/artc_2026/mapocc && python -c "import os; print(os.path.exists(\"data/nuscenes/bevdetv2-nuscenes_infos_train.pkl\")); print(os.path.exists(\"data/nuscenes/bevdetv2-nuscenes_infos_val.pkl\"))"'
```

預期輸出：

```bash
True
True
```

### Step M. Nano4 container 腳本的 bind 設定

舊 TWCC 腳本直接在 host 環境跑，例如 `TWCC/train_multi_cnn_head_map_neck.sh`，所以不用寫 container bind。

Nano4 腳本是在 Singularity container 裡跑。container 預設看不到所有 host 路徑，必須明確 bind 需要的目錄。

```bash
SINGULARITY_BIND_ARGS=(
    --bind /home/u2336262:/home/u2336262
    --bind /work:/work
)

if [ -n "${EXTRA_BINDS:-}" ]; then
    SINGULARITY_BIND_ARGS+=(--bind "${EXTRA_BINDS}")
fi

singularity exec --cleanenv --nv \
    "${SINGULARITY_BIND_ARGS[@]}" \
    "${SIF}" \
    bash -c '
```

用途：

- `--bind /home/u2336262:/home/u2336262`：讓 container 看得到 repo、conda env、SIF 以外的 home 目錄檔案。
- `--bind /work:/work`：讓 container 看得到 `mapocc/data -> /work/u2336262/data` 指到的資料。
- `EXTRA_BINDS`：保留給臨時追加其他資料路徑，例如 `/project:/project`。
- `"${SINGULARITY_BIND_ARGS[@]}"`：把上面所有 bind 參數傳給 `singularity exec`。
- `--cleanenv`：避免 host shell 的 Python / CUDA / module 環境污染 container。
- `--nv`：把主機 NVIDIA driver 與 GPU 掛進 container。

### Step N. 送 Nano4 test job

`dev` 是 H200 GPU test partition。`ngstest` 沒有 GPU，不要用。

```bash
cd /home/u2336262/Desktop/artc_2026/mapocc/TWCC_nano4

EXTRA_BINDS=/work:/work sbatch test_singularity_unimapocc_nano4.sh
```

### Step O. 送正式訓練

```bash
cd /home/u2336262/Desktop/artc_2026/mapocc/TWCC_nano4

EXTRA_BINDS=/work:/work sbatch train_fpn_lateral_aspp_focal_dice_weighted_nano4.sh
```
