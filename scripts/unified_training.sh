#!/bin/bash
set -e
export BNB_CUDA_VERSION=118


# export C_INCLUDE_PATH=/usr/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=/usr/include:$CPLUS_INCLUDE_PATH

echo "Starting unified symbol training..."
echo "Python path: $(which python)"
echo "Environment variables:"
env | grep -E "(RUN_NAME|dataset_type|model_type|use_output_mlp|hidden_dim|bypass_mlp)" | sort

# Build COMMON_ARGS like in unified_inference.sh
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
    --run_name \"${RUN_NAME}\""

# Add boolean flags only if true (like in unified_inference.sh)
if [ "$use_output_mlp" = "True" ] || [ "$use_output_mlp" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --use_output_mlp"
fi

if [ "$bypass_mlp" = "True" ] || [ "$bypass_mlp" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --bypass_mlp"
fi

echo "COMMON_ARGS: ${COMMON_ARGS}"

# Run the training script using eval like in unified_inference.sh
eval "python ${SCRIPT_PATH} ${COMMON_ARGS}"

echo "Training completed successfully!"