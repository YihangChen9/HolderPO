
model="oat-output/holder_7B/saved_models/step_00500/"

# Default: greedy n=1 for the standard suite; aime25 overridden to pass@8 (t=0.7, n=8).
# Uses evaluation_suite_v2 (built by prepare_aime25).
python utils/evaluation/evaluate_model.py \
    --model_name $model \
    --dataset_name ./datasets/evaluation_suite_v2 \
    --tasks '["aime","aime25","amc","math","minerva","olympiad_bench"]' \
    --n_samples 1 \
    --temperature 0 \
    --top_p 1 \
    --max_tokens 4096 \
    --max_model_len 5120 \
    --pass_n_tasks '["aime25"]' \
    --pass_n_samples 8 \
    --pass_temperature 0.7 \
    --metric pass
