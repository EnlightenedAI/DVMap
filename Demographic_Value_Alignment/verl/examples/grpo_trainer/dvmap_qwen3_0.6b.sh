# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

export PYTHONASYNCIODEBUG=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export HYDRA_FULL_ERROR=1 

export CUDA_VISIBLE_DEVICES=5,6
export RAY_LOG_LEVEL=INFO


rewards_name="wvs"
model_name="Qwen3-0.6B"
mkdir -p $RAY_TMPDIR
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=#PATH/train.parquet \
    data.val_files=#PATH/val.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="/data/pyzhu/models/Qwen/${model_name}" \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64\
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_fused_kernels=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    reward_model.model.use_fused_kernels=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    data.reward_fn_key=${rewards_name} \
    trainer.project_name='verl_grpo_wvs' \
    trainer.experiment_name="${model_name}_grpo_${rewards_name}" \
    trainer.n_gpus_per_node=1 \
    +reward_fn.score_method=custom1 \
    trainer.nnodes=2 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="#PATH/${model_name}_grpo_${rewards_name}/"  $@

    