export HF_TOKEN=TODO
export CACHE_DIR=TODO
export ATTN_IMPLEMENTATION='flash_attention_2'
export WANDB_API_KEY=TODO

CUDA_DEVICE_ID=2,3

dataset_name=aime2025
model_name="google/gemma-3-27b-it"
model_pretty_name="gemma27b"

# dataset_name=math500
# model_name="meta-llama/Llama-3.1-8B-Instruct"
# model_pretty_name="llama8b"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python gepa-run.py \
  --model_name ${model_name} \
  --dataset ${dataset_name} \
  --max_metric_calls 100 \
  --reflection_minibatch_size 3 \
  --use_wandb \
  --wandb_project gepa-${dataset_name} \
  --wandb_run_name ${model_name}-${dataset_name}-gepa \
  --train_full_size 20

