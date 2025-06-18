#!/usr/bin/env bash

# Default variables
base_model_id="Qwen/Qwen3-8B"
assistant_model_id="Qwen/Qwen3-0.6B"
params_file_name="None"
base_model_ckpt="None"
assistant_model_ckpt="None"
num_thought_tokens=0
num_return_sequences=1
num_scaling_times=1
task_name="gsm8k"
seed_from=41
seed_to=50
test_k=0
tune_assistant_model=false
print_input=false
print_response=false
model_name="soft"
log_dir="inference_logs"
run_name=""

# Argument parsing
while [[ $# -gt 0 ]]; do
    case ${1} in
        --base_model_id) base_model_id="${2}"; shift 2 ;;
        --assistant_model_id) assistant_model_id="${2}"; shift 2 ;;
        --params_file_name) params_file_name="${2}"; shift 2 ;;
        --assistant_model_ckpt) assistant_model_ckpt="${2}"; shift 2 ;;
        --base_model_ckpt) base_model_ckpt="${2}"; shift 2 ;;
        --num_thought_tokens) num_thought_tokens="${2}"; shift 2 ;;
        --num_return_sequences) num_return_sequences="${2}"; shift 2 ;;
        --num_scaling_times) num_scaling_times="${2}"; shift 2 ;;
        --task_name) task_name="${2}"; shift 2 ;;
        --model_name) model_name="${2}"; shift 2 ;;
        --seed_from) seed_from="${2}"; shift 2 ;;
        --seed_to) seed_to="${2}"; shift 2 ;;
        --test_k) test_k="${2}"; shift 2 ;;
        --tune_assistant_model) tune_assistant_model=true; shift ;;
        --print_input) print_input=true; shift ;;
        --print_response) print_response=true; shift ;;
        --log_dir) log_dir="${2}"; shift 2 ;;
        --run_name) run_name="${2}"; shift 2 ;;
        *) echo "Unknown argument: ${1}"; shift ;;
    esac
done

# Display configuration
echo "Script started at: $(date)"
echo "Base Model ID: ${base_model_id}"
echo "Assistant Model ID: ${assistant_model_id}"
echo "Task Name: ${task_name}"
echo "Model Name: ${model_name}"
echo "Num Thought Tokens: ${num_thought_tokens}"
echo "Num Return Sequences: ${num_return_sequences}"
echo "Num Scaling Times: ${num_scaling_times}"
echo "Seed Range: ${seed_from} to ${seed_to}"
echo "Test K: ${test_k}"
echo "Tune Assistant Model: ${tune_assistant_model}"
echo "Print Input: ${print_input}"
echo "Print Response: ${print_response}"
echo "Logs will be saved in: ${log_dir}"
echo "Run Name is : <|start|>${run_name}<|end|>"
echo "----------------------------------------"

base_model_name="${base_model_id#*/}"
assistant_model_name="${assistant_model_id#*/}"

# Ensure log directory exists
mkdir -p "${log_dir}"

# Check if the Python script exists
if [[ ! -f "evaluate_softcotpp.py" ]]; then
    echo "Error: evaluate_softcotpp.py not found!"
    exit 1
fi

# Loop over seed range
for ((seed=seed_from; seed<=seed_to; seed++)); do
    start_time=$(date +%s)
    echo "Running evaluation for seed: ${seed} at $(date)"

    log_file_name="${log_dir}/${run_name}${task_name}-${model_name}-seed_${seed}-ntt_${num_thought_tokens}-nrs_${num_return_sequences}-nst_${num_scaling_times}-${base_model_name}-${assistant_model_name}.log"

    cmd="python evaluate_softcotpp.py \
--base_model_id \"${base_model_id}\" \
--assistant_model_id \"${assistant_model_id}\" \
--params_file_name \"${params_file_name}\" \
--assistant_model_ckpt \"${assistant_model_ckpt}\" \
--base_model_ckpt \"${base_model_ckpt}\" \
--num_thought_tokens ${num_thought_tokens} \
--num_return_sequences ${num_return_sequences} \
--num_scaling_times ${num_scaling_times} \
--task_name \"${task_name}\" \
--model_name \"${model_name}\" \
--seed ${seed} \
--test_k ${test_k}"

    # Add optional flags
    ${tune_assistant_model} && cmd+=" --tune_assistant_model"
    ${print_input} && cmd+=" --print_input"
    ${print_response} && cmd+=" --print_response"

    # Run the command and redirect output
    echo "${cmd} > \"${log_file_name}\""
    eval "${cmd} > \"${log_file_name}\""

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    echo "Evaluation for seed ${seed} finished at: $(date)"
    echo "Elapsed time: ${elapsed_time} seconds"
    echo "========================================"
done

# Display the script end time
echo "Script finished at: $(date)"
