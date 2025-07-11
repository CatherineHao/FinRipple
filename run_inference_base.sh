#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

PYTHON_SCRIPT=""
PROMPT_BASEPATH=""
OUTPUT_BASEPATH=""
ADAPTER_BASEPATH=""

declare -A MODEL_PATHS=(

)

run_inference() {
    local base_model_path="$1"
    local adapter_path="$2"
    local prompt_path="$3"
    local model_type="$4"
    local output_path="$5"
    
    local model_name=$(basename "$base_model_path")
    local prompt_name=$(basename "$prompt_path" .csv)
    local log_file="${output_path}/${prompt_name}_${model_name}.log"
    
    echo "Running inference for $model_name with $prompt_name"
    python $PYTHON_SCRIPT "$base_model_path" "$adapter_path" "$prompt_path" "$model_type" "$output_path" "False" > "$log_file" 2>&1
    echo "Finished. Log saved to $log_file"
}

for base_model_path in "${!MODEL_PATHS[@]}"; do
    model_type="${MODEL_PATHS[$base_model_path]}"
    model_name=$(basename "$base_model_path")
    output_path="${OUTPUT_BASEPATH}/${model_name}"
    mkdir -p "$output_path"

    for prompt_path in ${PROMPT_BASEPATH}/merged_*_*_result_select_20.csv; do
        prompt_name=$(basename "$prompt_path" .csv)
        adapter_folder=$(echo "$prompt_name" | sed 's/_result_select_20$//')
        adapter_path="${ADAPTER_BASEPATH}/${model_name}/${adapter_folder}/checkpoint-30"

        if [ -d "$adapter_path" ]; then
            run_inference "$base_model_path" "$adapter_path" "$prompt_path" "$model_type" "$output_path"
        else
            echo "Warning: Adapter path not found: $adapter_path"
        fi
    done
done