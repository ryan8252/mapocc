#!/bin/bash

#SBATCH -J your_job_name                    # 任務名稱 (隨便取)
#SBATCH --account=your_project_account      # 計畫帳號 (從教授的帳號中查)
#SBATCH -p gp4d                             # 用可跑 4 天的分區
#SBATCH -N 1                                # 申請 1 台主機
#SBATCH --ntasks-per-node=8                 # 建議與 -N 一起使用，代表每台機器 8 個任務
#SBATCH --gres=gpu:8                        # 申請 8 顆 V100 GPU
#SBATCH --cpus-per-task=4                   # 每一顆 GPU 配 4 核 CPU
#SBATCH --mem=64G                           # 申請 64GB 系統記憶體
#SBATCH -o %j.log                           # 訓練 Log 輸出位置
#SBATCH -e %j.log                           # 錯誤 Log 輸出位置

source /home/u2336262/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name
cd to/your/project/path
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
bash your-training-script.sh