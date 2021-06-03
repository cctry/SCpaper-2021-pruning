# batch size tests:
NUM_EPOCHS=50
ADMM_TRAINING_LR=1.0
RETRAINING_LR=3.0
BLOCK_ROW_WIDTH=16
BLOCK_COL_WIDTH=16
BLOCK_SIZE=256
PRUNE_CONFIG=config_transformer_v51
SPARSITY_TYPE=whole_block_padding_balanced
# ADMM training:
CUDA_VISIBLE_DEVICES=1 python 12_Transformer_pruning_test_whole_block.py --admm_transformer --sparsity_type ${SPARSITY_TYPE} --block_size ${BLOCK_SIZE} --block_row_width ${BLOCK_ROW_WIDTH} --block_col_width ${BLOCK_COL_WIDTH} --batch_size 800 --load_model transformer_model_lr_3.0_50.pt --combine_progressive --epochs ${NUM_EPOCHS} --rho 0.001 --rho_num 1 --lr ${ADMM_TRAINING_LR} --lr_decay 20 --config_file ${PRUNE_CONFIG} && # -> !
# Retraining:
CUDA_VISIBLE_DEVICES=1 python 12_Transformer_pruning_test_whole_block.py --masked_retrain --sparsity_type ${SPARSITY_TYPE} --block_size ${BLOCK_SIZE} --block_row_width ${BLOCK_ROW_WIDTH} --block_col_width ${BLOCK_COL_WIDTH} --batch_size 800 --load_model transformer_model_admm_training_0_th_rho_0.001_${PRUNE_CONFIG}_${SPARSITY_TYPE}_${NUM_EPOCHS}_${BLOCK_SIZE}_${ADMM_TRAINING_LR}.pt --combine_progressive --epochs ${NUM_EPOCHS} --lr ${RETRAINING_LR} --lr_decay 20 --config_file ${PRUNE_CONFIG}
