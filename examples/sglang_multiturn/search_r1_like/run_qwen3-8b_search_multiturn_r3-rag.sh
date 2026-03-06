set -x

ulimit -n 65535

PROJECT_DIR=path/to/project
source ~/miniconda3/etc/profile.d/conda.sh

conda activate retriever
python $PROJECT_DIR/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py --faiss_gpu &
sleep 120


conda activate verl

TRAIN_DATA=$PROJECT_DIR/data/search-r1/train.parquet
VAL_DATA=$PROJECT_DIR/data/search-r1/filtered_test.parquet
CONFIG_PATH=$PROJECT_DIR/verl/trainer/config
TOOL_CONFIG=$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml

wandb_project_name=searchqa
wandb_experiment_name=searchqa-dense-r3-rag-qwen3-8b
base_model_path=/path/to/qwen3-8b
model_save_path=$PROJECT_DIR/checkpoints/$wandb_experiment_name-$(date +%m%d%H%M)
mkdir -p ${model_save_path}

CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline WANDB_DIR=$PROJECT_DIR python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    algorithm.adv_estimator=gae-round \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.return_raw_chat=True \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    reward_model.reward_manager=r3-rag \
    +reward_model.reward_kwargs.max_llm_judge_workers=128 \
    actor_rollout_ref.model.path=$base_model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=9e-6 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.use_remove_padding=True \
    critic.model.path=$base_model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.nccl_timeout=7200 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$wandb_project_name \
    trainer.experiment_name=$wandb_experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir=$model_save_path \
    trainer.rollout_data_dir=$model_save_path/rollout_data \
    trainer.validation_data_dir=$model_save_path/validation_data \
    2>&1 | tee $model_save_path/train.log
