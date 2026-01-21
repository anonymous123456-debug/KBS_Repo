#!/bin/bash
#SBATCH --job-name=ToGLamamcqa
#SBATCH --output=./analysis.out  # 输出文件，%A 是作业ID，%a 是数组索引（如果使用了数组作业）
#SBATCH --error=./analysis.err    # 错误文件
#SBATCH --nodes=1                  # 请求的节点数
#SBATCH --ntasks=1                # 请求的任务数（通常与节点数相匹配，除非使用了多核任务）
#SBATCH --cpus-per-task=8        # 每个任务请求的CPU数
#SBATCH --gres=gpu:2
#SBATCH --partition=A40


# 这里是你的作业命令
echo "Running job on node: $(hostname)"
# 替换下面的命令为你的实际作业命令
/share/home/ncu_418000240001/test/long/bin/python -u analysis.py --dataset commonsenseqa
date
hostname
