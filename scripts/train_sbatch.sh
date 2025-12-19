#!/bin/bash
#SBATCH --job-name=janusvln_train
#SBATCH --account=a144
#SBATCH --output=slurm-janusvln-train-%j.out
#SBATCH --error=slurm-janusvln-train-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --environment=/users/jiaqchen/.edf/faive2lerobot.toml
#SBATCH --requeue
#SBATCH --signal=USR1@600

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Check some specs
free -h
nvidia-smi --query-gpu=memory.total --format=csv

# Define environment vars for single-node training
export MASTER_ADDR=localhost
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
# Number of GPUs per node
export NPROC_PER_NODE=4

echo "[sbatch-master] MASTER_ADDR: $MASTER_ADDR"
echo "[sbatch-master] MASTER_PORT: $MASTER_PORT"
echo "[sbatch-master] NPROC_PER_NODE: $NPROC_PER_NODE"
echo "[sbatch-master] Training on single node with $NPROC_PER_NODE GPUs"

# Model and output paths
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
VGGT_MODEL_PATH="facebook/VGGT-1B"
OUTPUT_DIR="./JanusVLN_Base"
CACHE_DIR="./cache"
DATASETS="train_r2r_rxr"

mkdir -p $OUTPUT_DIR

echo "DATASETS: $DATASETS"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"

# NCCL configuration for single-node multi-GPU training
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1          # Disable InfiniBand
export NCCL_P2P_DISABLE=0         # Enable P2P (PCIe) for single-node
export NCCL_SHM_DISABLE=0         # Enable shared memory (important for single-node)
export NCCL_NET_GDR_LEVEL=0       # Disable GPU Direct RDMA
export NCCL_DEBUG=WARN
unset NCCL_NET
export NCCL_SOCKET_IFNAME=lo      # Use loopback for single-node

# DeepSpeed and training command
CMD="
source /users/jiaqchen/scratch/lsai_proj/env/lsai/bin/activate

echo 'Using Python: '\$(which python)
echo 'Python version: '\$(python --version)

cd /iopsstor/scratch/cscs/jiaqchen/lsai_proj/src/JanusVLN

python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --vggt_model_path $VGGT_MODEL_PATH \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --dataset_use $DATASETS \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 163840 \
    --data_flatten False \
    --max_pixels \$((576*28*28)) \
    --min_pixels \$((16*28*28)) \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels \$((1664*28*28)) \
    --video_min_frame_pixels \$((256*28*28)) \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --deepspeed scripts/zero3.json \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --group_by_modality_length true \
    --seed 42 \
    --report_to none \
    --reference_frame first
"

# Execute training directly on single node (torch.distributed.run will spawn 4 GPU processes)
bash -c "$CMD"

# Print completion information
echo "Training finished at: $(date)"

