set -e

ATTACK_MODEL="Qwen/Qwen2.5-0.5B"
REWARD_MODEL="EleutherAI/pythia-160m"
REWRITE_MODEL="/path/to/rewrite_model"
TARGET_MODEL="/path/to/target_model"  
DOWNGRADE_MODEL="/path/to/downgrade_model"
SGLANG_ENDPOINT="http://localhost:30000"

OUTPUT_DIR="./output/redteam_test"
BATCH_SIZE=2
GRAD_ACCUM=4
EPISODES=100


if ! curl -s "$SGLANG_ENDPOINT/health" > /dev/null 2>&1; then
    echo "Warning: SGLang endpoint not responding. Make sure it's running."
fi

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/redteam/train_redteam.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$ATTACK_MODEL" \
    --reward_model_path "$REWARD_MODEL" \
    --rewrite_model_path "$REWRITE_MODEL" \
    --target_model_path "$TARGET_MODEL" \
    --downgrade_model_path "$DOWNGRADE_MODEL" \
    --sglang_endpoint "$SGLANG_ENDPOINT" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_ppo_epochs 4 \
    --learning_rate 1e-6 \
    --response_length 128 \
    --total_episodes $EPISODES \
    --diversity_threshold 0.85 \
    --kl_coef 0.05 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 50 \
    --report_to none
