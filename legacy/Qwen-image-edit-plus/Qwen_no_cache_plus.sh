#!/bin/bash

NUM_SHARDS=4  # 总共切成 4 份，对应 4 组

script_dir=/home/dataset-local/chenxueqing/code/image-edit-round-reuse/CacheEdit/Qwen-image-edit-plus/Qwen_main_eval_parallel.py
MODEL_PATH=/home/dataset-local/chenxueqing/model/Qwen/Qwen-Image-Edit
DATA_PATH=/home/dataset-local/chenxueqing/datasets/test
OUT_ROOT=/home/dataset-local/chenxueqing/result/QwenImageEdit/kontext-bench-test/original

# 组 0：用 GPU 0,1
CUDA_VISIBLE_DEVICES=0 python ${script_dir} \
  --model_path ${MODEL_PATH} \
  --image_path ${DATA_PATH} \
  --output_dir ${OUT_ROOT}_shard0 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --true_cfg_scale 4.0 \
  --num_shards ${NUM_SHARDS} \
  --shard_index 0 &

# 组 1：用 GPU 2,3
CUDA_VISIBLE_DEVICES=1 python ${script_dir} \
  --model_path ${MODEL_PATH} \
  --image_path ${DATA_PATH} \
  --output_dir ${OUT_ROOT}_shard1 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --true_cfg_scale 4.0 \
  --num_shards ${NUM_SHARDS} \
  --shard_index 1 &

# 组 2：用 GPU 4,5
CUDA_VISIBLE_DEVICES=2 python ${script_dir} \
  --model_path ${MODEL_PATH} \
  --image_path ${DATA_PATH} \
  --output_dir ${OUT_ROOT}_shard2 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --true_cfg_scale 4.0 \
  --num_shards ${NUM_SHARDS} \
  --shard_index 2 &

# 组 3：用 GPU 6,7
CUDA_VISIBLE_DEVICES=3 python ${script_dir} \
  --model_path ${MODEL_PATH} \
  --image_path ${DATA_PATH} \
  --output_dir ${OUT_ROOT}_shard3 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --true_cfg_scale 4.0 \
  --num_shards ${NUM_SHARDS} \
  --shard_index 3 &

wait
echo "All 4 shards finished."
