#!/bin/bash
#SBATCH --job-name=roberta-flash-attention
#SBATCH --cpus-per-task=8 --mem=8000M
#SBATCH --exclude=hendrixgpu01fl
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --output=/home/rwg642/flash-roberta/roberta-flash-attention.txt
#SBATCH --time=0:20:00

module load miniconda/4.12.0
conda init bash
conda activate kiddothe2b

echo $SLURMD_NODENAME
echo $CUDA_VISIBLE_DEVICES

python demo_mlm.py --model_class roberta
echo "----------------------------------------"
python demo_mlm.py --model_class flash-roberta