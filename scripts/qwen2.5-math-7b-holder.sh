# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Keep system CUDA libraries untouched to avoid runtime conflicts.
# export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

N_GPU=4
N_SAMPLE=8
SAVE_STEPS=100
CLIPRANGE=0.2
HOLDER_P=3.0
HOLDER_P_SCHEDULE="linear_dec"   # constant | linear | linear_dec | sin | cos | quad | quad_dec | cubic | cubic_dec
HOLDER_P_POWER=2.0           # exponent n for quad / quad_dec
HOLDER_P_MIN=-2.0
HOLDER_P_MAX=2.0
HOLDER_P_SCHEDULE_STEPS=500
MODEL_NAME="deepseek-r1-distill-qwen-7b"
# wandb is opt-in: set USE_WB=1 (and export WANDB_API_KEY) to enable logging.
USE_WB=${USE_WB:-0}
WB_FLAGS=""
if [ "$USE_WB" = "1" ]; then
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "USE_WB=1 but WANDB_API_KEY is empty. Export it before running." >&2
    exit 1
  fi
  WB_FLAGS="--use-wb --wb_project holder --wb-run-name ${MODEL_NAME}-holder-${HOLDER_P_SCHEDULE}-[${HOLDER_P_MIN},${HOLDER_P_MAX}]"
fi

# Qwen-Math template
python train_zero_math_holder.py \
    --critic_type grpo \
    --holder_p $HOLDER_P \
    --holder_p_schedule $HOLDER_P_SCHEDULE \
    --holder_p_min $HOLDER_P_MIN \
    --holder_p_max $HOLDER_P_MAX \
    --holder_p_schedule_steps $HOLDER_P_SCHEDULE_STEPS \
    --holder_p_power $HOLDER_P_POWER \
    --gpus $N_GPU \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --cliprange $CLIPRANGE \
    --oracle_type reward \
    --oracle math \
    --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --prompt_template qwen_math \
    --verifier_version math_verify \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data understand_r1_zero_main/datasets/train/math_lvl3to5_8k \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 20 \
    --prompt_max_length 1024 \
    --num_samples $N_SAMPLE \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --save_steps $SAVE_STEPS \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device $((128 / N_GPU)) \
    --pi_buffer_maxlen_per_device $((128 * N_SAMPLE / N_GPU)) \
    --eval_batch_size 200 \
    --eval_steps 16 \
    --eval_temperature 0 \
    --eval_top_p 1 \
    --eval_n 1 \
    --eval_pass_n_tasks aime25 \
    --eval_pass_n 8 \
    --eval_pass_temperature 0.7 \
    --eval_pass_top_p 1 \
    --eval_generate_max_length 4096 \
    --eval_data ./datasets/evaluation_suite_v2 \
    --eval_input_key input \
    $WB_FLAGS \
    --critic_type_modify holder