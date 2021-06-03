export GLUE_DIR=data/glue
export TASK_NAME=MRPC
export PENALTY=penalty_bert-4
export PRUNE_RATIO=bert_prune_ratios_30
export IN_DIR=data/prune/MRPC/whole_block_padding
export OUT_DIR=data/retrain/MRPC/whole_block_padding

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path $IN_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --masked_retrain \
  --lr_retrain 5e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --num_train_epochs 4.0 \
  --sparsity_type whole_block_padding \
  --evaluate_during_training \
  --block_row_division 3 \
  --block_row_width 3 \
  --output_dir $OUT_DIR \
  --overwrite_output_dir \
  --logging_steps 1500 \
  --logging_dir $OUT_DIR \
  --save_steps 4000
