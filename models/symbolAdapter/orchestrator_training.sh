#!/bin/bash
set -e

echo "Starting Orchestrator Symbol Training..."
echo "Python path: $(which python)"
echo "Environment variables:"
env | grep -E "(RUN_NAME|dataset_type|model_type|schedule_type|use_output_mlp|bypass_mlp|dynamic_symbols_per_epoch)" | sort

# Script path
# SCRIPT_PATH="/home/neeraja/code/ICL/models/symbolAdapter/orchestrator_training.py"

# Build COMMON_ARGS
COMMON_ARGS="--model_type \"${model_type}\" \
    --dataset_type \"${dataset_type}\" \
    --device \"${device}\" \
    --lora_lr ${lora_lr} \
    --mlp_lr ${mlp_lr} \
    --lora_epochs ${lora_epochs} \
    --mlp_epochs ${mlp_epochs} \
    --total_cycles ${total_cycles} \
    --lora_final_epochs ${lora_final_epochs} \
    --hidden_dim ${hidden_dim} \
    --batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_grad_norm ${max_grad_norm} \
    --max_samples ${max_samples} \
    --output_dir \"${OUTPUT_DIR}\" \
    --run_name \"${RUN_NAME}\" \
    --schedule_type \"${schedule_type}\""

# Add boolean flags only if true
if [ "$use_output_mlp" = "True" ] || [ "$use_output_mlp" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --use_output_mlp"
fi

if [ "$bypass_mlp" = "True" ] || [ "$bypass_mlp" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --bypass_mlp"
fi

if [ "$dynamic_symbols_per_epoch" = "True" ] || [ "$dynamic_symbols_per_epoch" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --dynamic_symbols_per_epoch"
fi

echo "COMMON_ARGS: ${COMMON_ARGS}"

# Run the training script using eval
eval "python ${SCRIPT_PATH} ${COMMON_ARGS}"

echo "Orchestrator training completed successfully!"