export GLUE_DIR=data/glue
export TASK_NAME=RTE


CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-13 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty13 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty13 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-14 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty14 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty14 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-15 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty15 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty15 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-16 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty16 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty16 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-17 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty17 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty17 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-18 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty18 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty18 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-19 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty19 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty19 \
  --save_steps 4000  



