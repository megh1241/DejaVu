file=/home/ubuntu/DejaVu/Decentralized_FM_alpha/c4_train/c4_train.jsonl
output_file=/home/ubuntu/DejaVu/Decentralized_FM_alpha/c4_train/output_c4_train.jsonl

export PATH_TO_MODEL_CHECKPOINT=/home/ubuntu/DejaVu/Decentralized_FM_alpha/pretrained_models
echo "start running ${file}"
export SPRARSE_PATH=/home/ubuntu/DejaVu/sparse_predictor/pred_models
export LAYER=86
export ATTN_TOPK_1=24
export ATTN_TOPK_2=48
export SPARSE_ATT=1

LAYER=86
export TOPK=1024
ATTN_TOPK_1=24
ATTN_TOPK_2=48

ARGS="--model-name $PATH_TO_MODEL_CHECKPOINT \
--model-type opt-ml-att-sparse \
--seed 42 \
--fp16 \
--num-layers 3 \
--max-layers 24 \
--budget 10800 \
--num-iters 1000 \
--dist-url tcp://127.0.0.1:9969 \
--token-micro-batch-size 2 \
--world-size 8 --pipeline-group-size 8 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    &
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)
