export GLUE_DIR=data/glue
export TASK_NAME=MRPC
export INT_DIR=bert-base-uncased
export OUT_DIR=data/finetune/MRPC

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
          --model_name_or_path $INT_DIR \
          --task_name $TASK_NAME \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 384 \
          --per_gpu_train_batch_size 16 \
          --per_gpu_eval_batch_size 16 \
          --learning_rate 5e-5 \
          --num_train_epochs 4.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 400 \
          --logging_dir $OUT_DIR \
          --save_steps 4000
