export GLUE_DIR=data/glue
export TASK_NAME=MRPC
export PENALTY=penalty_bert-4
export INT_DIR=data/finetune/MRPC
export OUT_DIR=data/prune/MRPC/whole_block_padding


CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path $INT_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file ${PENALTY} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --evaluate_during_training \
  --sparsity_type whole_block_padding \
  --block_row_division 3 \
  --block_row_width 3 \
  --output_dir $OUT_DIR \
  --overwrite_output_dir \
  --logging_steps 2000 \
  --logging_dir $OUT_DIR \
  --save_steps 4000
