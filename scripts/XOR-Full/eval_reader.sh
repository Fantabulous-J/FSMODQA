MODEL_PATH=checkpoints/fsmodqa-600k/checkpoint-best

python test_reader.py \
  --output_dir ${MODEL_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --output_path dev_reader_xor_eng_span_predictions.json \
  --save_steps 500 \
  --train_dir data/XOR-Full \
  --train_path ${MODEL_PATH}/dev_xor_retrieve_pids.jsonl \
  --corpus_file all_w100.tsv \
  --query_file xor_dev_full_v1_1.jsonl \
  --per_device_eval_batch_size 1 \
  --ddp_find_unused_parameters False \
  --train_n_passages 100 \
  --max_query_length 50 \
  --max_passage_length 200 \
  --max_query_passage_length 250 \
  --max_answer_length 50 \
  --separate_joint_encoding \
  --de_avg_pooling \
  --add_lang_token \
  --bf16 False \
  --tf32 True