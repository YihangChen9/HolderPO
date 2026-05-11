set -x
ENGINE=${1:-vllm}
VARIANT=${2:-sequence}
HOLDER_P=${HOLDER_P:-1.0}
HOLDER_P_SCHEDULE=${HOLDER_P_SCHEDULE:-constant}
HOLDER_P_MIN=${HOLDER_P_MIN:--2.0}
HOLDER_P_MAX=${HOLDER_P_MAX:-2.0}
HOLDER_P_SCHEDULE_STEPS=${HOLDER_P_SCHEDULE_STEPS:-1000}
HOLDER_P_POWER=${HOLDER_P_POWER:-2.0}
EXTRA_ARGS=("${@:3}")
export VLLM_ATTENTION_BACKEND=XFORMERS

case "$VARIANT" in
    sequence)
        LOSS_MODE="holder_seq"
        EXPERIMENT_SUFFIX="sequence"
        ;;
    token)
        LOSS_MODE="holder_token"
        EXPERIMENT_SUFFIX="token"
        ;;
    *)
        echo "Unknown Holder-PO variant: $VARIANT. Expected 'sequence' or 'token'." >&2
        exit 1
        ;;
esac

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
    actor_rollout_ref.actor.policy_loss.holder_p=$HOLDER_P \
    actor_rollout_ref.actor.policy_loss.holder_p_schedule=$HOLDER_P_SCHEDULE \
    actor_rollout_ref.actor.policy_loss.holder_p_min=$HOLDER_P_MIN \
    actor_rollout_ref.actor.policy_loss.holder_p_max=$HOLDER_P_MAX \
    actor_rollout_ref.actor.policy_loss.holder_p_schedule_steps=$HOLDER_P_SCHEDULE_STEPS \
    actor_rollout_ref.actor.policy_loss.holder_p_power=$HOLDER_P_POWER \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name="holder_${EXPERIMENT_SUFFIX}_qwen2.5_1.5b" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True "${EXTRA_ARGS[@]}"
