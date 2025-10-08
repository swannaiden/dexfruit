
ALG="train_xarm_dt_2d_timm"
TAG="exp_name"
DEBUG="false"
SEED="42"
WANDB_MODE="online"  # options: "online", "offline", "disabled"
PORT="25909"

NUM_GPUS=$(grep -oP '#SBATCH --gres=gpu:\K\d+' "$0" || echo "1")
echo "Using $NUM_GPUS GPUs"


EXP="${TAG}"
RUN_DIR="runs/${EXP}"

accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port ${PORT} train.py \
  --config-name "${ALG}.yaml" \
  hydra.run.dir="${RUN_DIR}" \
  training.debug="${DEBUG}" \
  training.seed="${SEED}" \
  exp_name="${EXP}" \
  logging.mode="${WANDB_MODE}"

echo "Done"
