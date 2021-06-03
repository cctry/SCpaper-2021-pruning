from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import operator
import random

from testers import *
from numpy import linalg as LA
import yaml

import numpy as np

class ADMM:
    def __init__(self, model, file_name, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        self.init(file_name, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream)
                self.prune_ratios = raw_dict['prune_ratios']
                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho
                for (name, W) in model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
            except yaml.YAMLError as exc:
                print(exc)


def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

# row pruning for each entire block of rows: -libn
def block_rows_pruning(args, block_weight_np, prune_ratio):

    # Devide each block of rows into several small blocks based on column: -libn
    row_l2_norm = []
    block_weight_np_backup = block_weight_np.copy()
    # Step 1: divide weight matrix into blocks:
    org_shape = block_weight_np.shape
    # group_size_columns = 41      # block size -libn
    group_size_columns = args.block_size    # block size -libn
    org_cols = org_shape[1]        #全部的列数
    remain_cols = org_cols%group_size_columns   #被block_prune后，剩下的列数
    group_wt_org_shape = block_weight_np[:,:(org_cols-remain_cols)].shape   # 被block_prune的所有列数
    if remain_cols == 0:
        weight_groups = block_weight_np.reshape((-1, org_shape[0], group_size_columns))
        zero_rows = np.zeros((org_shape[0], weight_groups.shape[0]))
    else:
        weight_groups = block_weight_np[:,:(org_cols-remain_cols)].reshape((-1, org_shape[0], group_size_columns))
        zero_rows = np.zeros((org_shape[0], weight_groups.shape[0]+1))
    # weight_groups = weight.reshape((-1, group_size_columns, org_shape[1]))
    groups_shape = weight_groups.shape
    group_mask = np.zeros(groups_shape, dtype=np.float32)
    percent = prune_ratio * 100
    for gp in range(groups_shape[0]):
        # for each small block (weight_groups[gp]): -libn
        # Step 2: prune each block using column pruning:
        # group_mask[gp, :, :], weight_groups[gp, :, :] = rows_pruning(args, weight_groups[gp], prune_ratio)


        # L2 row pruning:
        row_l2_norm = LA.norm(weight_groups[gp], 2, axis=1)     # calculate the norm of each row!
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight_groups[gp, under_threshold, :] = 0
        zero_rows[under_threshold,gp] = 1
        above_threshold = above_threshold.astype(np.float32)


        for i in range(groups_shape[1]):    # groups_shape[1]: height of each small block. -libn
            group_mask[gp, i, :] = above_threshold[i]
    above_threshold_msk = group_mask.reshape(group_wt_org_shape)
    # above_threshold_msk = above_threshold_msk.reshape(org_shape)
    weight_groups = weight_groups.reshape(group_wt_org_shape)

    if remain_cols != 0:
        group_cols = org_cols-remain_cols
        weight_remain = block_weight_np[:,group_cols:]

        # for the remained rows (weight_remain): -libn
        # L2 row pruning:
        row_l2_norm = LA.norm(weight_remain, 2, axis=1)     # calculate the norm of each row!
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold_remain = row_l2_norm < percentile
        above_threshold_remain = row_l2_norm > percentile
        weight_remain[under_threshold_remain, :] = 0
        zero_rows[under_threshold,-1] = 1
        above_threshold_remain = above_threshold_remain.astype(np.float32)


        remain_shape = weight_remain.shape
        # Step 2: prune each block using block pruning:
        # above_threshold_remain, weight_remain = block_rows_pruning(args, weight_remain, prune_ratio)

        # column_l2_norm = LA.norm(weight_remain, 2, axis=0)
        # percentile = np.percentile(column_l2_norm, percent)
        # under_threshold = column_l2_norm < percentile
        # above_threshold = column_l2_norm > percentile
        # weight_remain[:, under_threshold] = 0
        remain_mask = np.zeros(remain_shape, dtype=np.float32)
        for i in range(weight_remain.shape[0]):
            remain_mask[i, :] = above_threshold_remain[i]
        # remain_mask = remain_mask.astype(np.float32)
        block_weight_np = np.concatenate((weight_groups, weight_remain), axis=1)
        above_threshold_msk = np.concatenate((above_threshold_msk, remain_mask), axis=1)
    else:
        block_weight_np = weight_groups

    # Step 3: combine all small blocks & avoid whole-row removement: -libn
    for i in range(zero_rows.shape[0]):
        if zero_rows[i,:].max() == 0:
            # print('%d th row: whole-row removement avoided!' %i)
            block_weight_np[i,:] = block_weight_np_backup[i,:]


    return above_threshold_msk, block_weight_np


def weight_pruning(args, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (args.sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0

        # pruning results display:
        print('Max(abs()) before pruning: %.3f' %(abs(weight_temp).min()))
        print('Max(abs()) after pruning: %.3f' %(abs(weight).min()))

        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "column"):

        # pruning results display:
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values


        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)

        # pruning results display:
        print('Max(abs()) before pruning: %.3f' %(abs(weight_temp).min()))
        print('Max(abs()) after pruning: %.3f' %(abs(weight).min()))

        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "whole_block_padding"): # -libn
        # shape = weight.shape
        shape = weight.shape
        conv = weight.reshape(shape[0],-1)  # type(conv) = np.array
        shape = conv.shape

        # Square blocks:
        block_row_width = args.block_row_width
        block_col_width = args.block_col_width
        # block_row_width = args.block_size#block_sizes[0]
        # block_col_width = args.block_size#block_sizes[1]
        print("Block size: %d, %d" %(block_row_width,block_col_width))


        if conv.shape[1]%block_col_width != 0 or conv.shape[0]%block_row_width != 0:
            print("the layer size is not divisible by block_col_width:",conv.shape[0], conv.shape[1], block_row_width, block_col_width)
            padding_height = (block_row_width-conv.shape[0]%block_row_width)%(block_row_width)
            padding_width = (block_col_width-conv.shape[1]%block_col_width)%(block_col_width)
            print("padding: height: %d; width: %d" %(padding_height, padding_width))
            conv = np.concatenate((conv, np.zeros((padding_height, conv.shape[1])).astype(conv.dtype)), axis=0)
            conv = np.concatenate((conv, np.zeros((conv.shape[0], padding_width)).astype(conv.dtype)), axis=1)

        block_col_division = int(conv.shape[1]/block_col_width)


        # Step 1: column division:
        # Divide the weight matrix into several blocks according to column
        convfrag = torch.chunk(torch.tensor(conv), block_col_division, dim=1)
        # Concatenate the derived blocks
        convfrag = torch.cat(convfrag, 0)
        # Calculate row l2 norm:
        block_row_norms = torch.norm(convfrag, dim=1)


        # Step 2: row division:
        # Divide the weight matrix into several blocks according to row
        block_row_division = int(len(block_row_norms)/block_row_width)
        block_norms = torch.chunk(block_row_norms.reshape(1,len(block_row_norms)), block_row_division, dim=1)
        # Concatenate the row l2 norms:
        block_norms = torch.cat(block_norms, 0)
        block_norms = torch.norm(block_norms, dim=1)

        percentile = np.percentile(block_norms, percent)  # get a value for this percentitle
        above_threshold = block_norms > percentile
        # # # revise above_threshold value to achieve balance:
        # # # above_threshold_before_balanced_pruning:
        above_threshold_before_BP = np.array(np.transpose(block_norms.reshape([-1, int(conv.shape[0]/block_row_width)])))
        # above_threshold_balanced = np.percentile(above_threshold_before_BP, percent, axis=1) # obtain the percentile for every row of the above_threshold_before_BP. -libn
        # above_threshold_after_BP = above_threshold_before_BP > above_threshold_balanced.reshape([-1,1])
        # above_threshold = torch.tensor(above_threshold_after_BP.reshape(block_norms.shape))


        # Step 3: to reproduce weight matrix using above_threshold:
        above_threshold_matrix = np.zeros(conv.shape).astype('int')
        for kk in range(len(above_threshold)):
            row_start = int(np.floor(kk/above_threshold_before_BP.shape[1])*block_row_width)
            col_start = int((kk%above_threshold_before_BP.shape[1])*block_col_width)
            # row_start = (kk%block_row_width)*block_row_width
            # col_start = int(block_col_width*(np.floor(kk/block_row_width)))
            # print("row_start, col_start:", row_start, col_start, ";value:", above_threshold[kk])
            above_threshold_matrix[row_start:row_start+block_row_width, col_start:col_start+block_col_width] = above_threshold[kk].numpy() * np.ones((block_row_width,block_col_width)).astype('int')
        # above_threshold_tmp = torch.chunk(torch.tensor(above_threshold_chunked_matrix), block_col_division, dim=0)
        # above_threshold_matrix = torch.cat(above_threshold_tmp, 1).numpy()

        conv *= above_threshold_matrix

        weight = conv[:shape[0],:shape[1]].reshape(weight.shape)
        above_threshold_matrix = above_threshold_matrix[:shape[0],:shape[1]].reshape(weight.shape)

        return torch.from_numpy(above_threshold_matrix.reshape(weight.shape)).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "whole_block_padding_balanced"): # -libn
        ori_shape = weight.shape
        conv = weight.reshape(ori_shape[0], -1)  # type(conv) = np.array
        shape = conv.shape
        block_row_width = args.block_row_width  # block_sizes[0]
        block_col_width = args.block_col_width  # block_sizes[1]

        # print(shape)
        if conv.shape[1] % block_col_width != 0 or conv.shape[0] % block_row_width != 0:
            padding_height = (block_row_width - conv.shape[0] % block_row_width) % (block_row_width)
            padding_width = (block_col_width - conv.shape[1] % block_col_width) % (block_col_width)
            conv = np.concatenate((conv, np.zeros((padding_height, conv.shape[1]))), axis=0)
            conv = np.concatenate((conv, np.zeros((conv.shape[0], padding_width)).astype(conv.dtype)), axis=1)
        # print(conv.shape)

        block_row_division = int(conv.shape[0] / block_row_width)
        block_col_division = int(conv.shape[1] / block_col_width)
        # print("block_row_division: ", block_row_division)
        # print("block_col_division: ", block_col_division)

        # whole_above_threshold = []
        above_threshold_matrix = np.zeros(conv.shape).astype('int')
        whole_above_threshold = torch.zeros(block_row_division * block_col_division)
        mask = np.zeros(conv.shape).astype('int')

        # print("whole_above_threshold: ", whole_above_threshold)
        for i in range(block_row_division):
            temp = conv[i * block_row_width: (i + 1) * block_row_width, :]
            # print(temp.shape)
            convfrag = torch.chunk(torch.tensor(temp), block_col_division, dim=1)
            # print(convfrag)
            convfrag = torch.cat(convfrag, 0)
            # print("convfrag.shape: ", convfrag.shape)

            block_row_norms = torch.norm(convfrag, dim=1)
            block_norms = torch.chunk(block_row_norms.reshape(1, len(block_row_norms)), block_col_division, dim=1)
            block_norms = torch.cat(block_norms, 0)
            block_norms = torch.norm(block_norms, dim=1)
            # print("block_norms: ", block_norms)
            percentile = np.percentile(block_norms, percent)
            above_threshold = block_norms > percentile
            # print("len of above_threshold: ", len(above_threshold))
            whole_above_threshold[i * block_col_division:i * block_col_division + block_col_division] = above_threshold[
                                                                                                        :]
            # print("above_threshold: ", above_threshold)
            # whole_above_threshold.append(above_threshold)
        # print("whole_above_threshold: ", whole_above_threshold)

        for kk in range(len(whole_above_threshold)):
            row_start = int(np.floor(kk / block_col_division) * block_row_width)
            col_start = int((kk % block_col_division) * block_col_width)
            mask[row_start:row_start + block_row_width, col_start:col_start + block_col_width] = whole_above_threshold[
                                                                                                     kk].numpy() * np.ones(
                (block_row_width, block_col_width)).astype('int')

        conv *= mask
        weight = conv[:shape[0], :shape[1]].reshape(weight.shape)
        # print(weight)
        mask = mask[:shape[0], :shape[1]].reshape(weight.shape)
        # print(type(weight))
        return torch.from_numpy(mask.reshape(weight.shape)).cuda(), torch.from_numpy(weight).float().cuda()
    else:
        raise SyntaxError("Unknown sparsity type")


def decoder_weight_pruning(args, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100



    if (args.sparsity_type == "whole_block_padding_balanced"):  # -libn
        weight = np.transpose(weight)
        ori_shape = weight.shape
        conv = weight.reshape(ori_shape[0], -1)  # type(conv) = np.array
        shape = conv.shape
        block_row_width = args.block_row_width  # block_sizes[0]
        block_col_width = args.block_col_width  # block_sizes[1]

        # print(shape)
        if conv.shape[1] % block_col_width != 0 or conv.shape[0] % block_row_width != 0:
            padding_height = (block_row_width - conv.shape[0] % block_row_width) % (block_row_width)
            padding_width = (block_col_width - conv.shape[1] % block_col_width) % (block_col_width)
            conv = np.concatenate((conv, np.zeros((padding_height, conv.shape[1]))), axis=0)
            conv = np.concatenate((conv, np.zeros((conv.shape[0], padding_width)).astype(conv.dtype)), axis=1)
        # print(conv.shape)

        block_row_division = int(conv.shape[0] / block_row_width)
        block_col_division = int(conv.shape[1] / block_col_width)

        above_threshold_matrix = np.zeros(conv.shape).astype('int')
        whole_above_threshold = torch.zeros(block_row_division * block_col_division)
        mask = np.zeros(conv.shape).astype('int')

        # print("whole_above_threshold: ", whole_above_threshold)
        for i in range(block_row_division):
            temp = conv[i * block_row_width: (i + 1) * block_row_width, :]
            # print(temp.shape)
            convfrag = torch.chunk(torch.tensor(temp), block_col_division, dim=1)
            # print(convfrag)
            convfrag = torch.cat(convfrag, 0)
            # print("convfrag.shape: ", convfrag.shape)

            block_row_norms = torch.norm(convfrag, dim=1)
            # print(len(block_row_norms))

            # block_row_division = int(len(block_row_norms)/block_row_width)
            # print("block_row_division", block_row_division)
            block_norms = torch.chunk(block_row_norms.reshape(1, len(block_row_norms)), block_col_division, dim=1)
            block_norms = torch.cat(block_norms, 0)
            block_norms = torch.norm(block_norms, dim=1)
            # print("block_norms: ", block_norms)
            percentile = np.percentile(block_norms, percent)
            above_threshold = block_norms > percentile
            # print("len of above_threshold: ", len(above_threshold))
            whole_above_threshold[i * block_col_division:i * block_col_division + block_col_division] = above_threshold[
                                                                                                        :]
            # print("above_threshold: ", above_threshold)
            # whole_above_threshold.append(above_threshold)
        # print("whole_above_threshold: ", whole_above_threshold)

        for kk in range(len(whole_above_threshold)):
            row_start = int(np.floor(kk / block_col_division) * block_row_width)
            col_start = int((kk % block_col_division) * block_col_width)
            mask[row_start:row_start + block_row_width, col_start:col_start + block_col_width] = whole_above_threshold[
                                                                                                     kk].numpy() * np.ones(
                (block_row_width, block_col_width)).astype('int')

        conv *= mask
        weight = conv[:shape[0], :shape[1]].reshape(weight.shape)
        # print(weight)
        mask = mask[:shape[0], :shape[1]].reshape(weight.shape)
        mask = np.transpose(mask)
        weight = np.transpose(weight)
        return torch.from_numpy(mask.reshape(weight.shape)).cuda(), torch.from_numpy(weight).float().cuda()
    else:
        raise SyntaxError("Unknown sparsity type")


def hard_prune(args, ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    print("hard pruning")
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None and name != "decoder.weight":
            _, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_ratios[name])  # get sparse model in cuda
        elif option == None and name == "decoder.weight":
            _, cuda_pruned_weights = decoder_weight_pruning(args, W, ADMM.prune_ratios[name])
        elif option == "random":
            _, cuda_pruned_weights = random_pruning(args, W, ADMM.prune_ratios[name])

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(args, W, ADMM.prune_ratios[name])
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable


def test_sparsity(args, ADMM, model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    total_zeros = 0
    total_nonzeros = 0
    if args.sparsity_type == "irregular":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(total_weight_number / total_nonzeros))
    elif args.sparsity_type == "column":
        for i, (name, W) in enumerate(model.named_parameters()):

            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            column_l2_norm = LA.norm(W2d, 2, axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("column sparsity of layer {} is {}".format(name, zero_column / (zero_column + nonzero_column)))
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif args.sparsity_type == "filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            row_l2_norm = LA.norm(W2d, 2, axis=1)
            zero_row = np.sum(row_l2_norm == 0)
            nonzero_row = np.sum(row_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("filter sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif args.sparsity_type == "bn_filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            nonzeros = np.sum(W != 0)
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))


def admm_initialization(args, ADMM, model):
    if not args.admm_transformer:
        return
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            _, updated_Z = weight_pruning(args, W, ADMM.prune_ratios[name])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer, current_rho_num):
    if not args.admm_transformer:
        return
    # print('z_u_updating!!!')
    if epoch != 1 and (epoch - 1) % args.admm_epoch == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            if (args.verbose):
                Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
            ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name],
                                          ADMM.prune_ratios[name])  # equivalent to Euclidean Projection

            # print('z_u_updating pruning !!!')

            ADMM.ADMM_Z[name] = updated_Z
            if (args.verbose):
                if writer:
                    writer.add_scalar('layer:{} W(k+1)-Z(k+1)'.format(name),
                                      torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item(), current_rho_num*args.epochs+epoch)
                    writer.add_scalar('layer:{} Z(k+1)-Z(k)'.format(name),
                                      torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev) ** 2)).item(), current_rho_num*args.epochs+epoch)
                # print ("at layer {}. W(k+1)-Z(k+1): {}".format(name,torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item()))
                # print ("at layer {}, Z(k+1)-Z(k): {}".format(name,torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item()))
            ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm_transformer:

        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_ratios:
                continue

            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = args.admm_epoch
    lr = None
    if epoch % admm_epoch == 0:
        lr = args.lr
    else:
        admm_epoch_offset = epoch % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.lr * (0.1 ** (admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training using ADMM with masks (loss_sum = loss(output,target)+loss(U,Z)). -libn
# Used for training: to update: Weight, U, Z according to loss(output, target)+loss(U,Z). -libn
def admm_masked_train(args, ADMM, model, device, train_loader, optimizer, epoch, TEXT, writer, current_rho_num):
    model.train()
    masks = {}

    # get masks from parameters! -libn
    for i, (name, W) in enumerate(model.named_parameters()):
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask

    if epoch == 1:
        # inialize Z variable
        # print("Start admm training quantized network, quantization type: {}".format(args.quant_type))
        admm_initialization(args, ADMM, model)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Modification 1: added by libn for transformer!!! ---- start ----   -libn
        data = data.reshape([data.shape[1],data.shape[2]])
        target = target.reshape(target.shape[1])
        # print('input.shape = ',data.shape)
        # print('target.shape = ',target.shape)
        # Modification 1: added by libn for transformer!!! ---- end ----    -libn
        optimizer.zero_grad()

        output = model(data)

        # ce_loss = F.cross_entropy(output, target)
        # Modification 2: replace the ce_loss function!!! ---- start ----   -libn
        ntokens = len(TEXT.vocab.stoi)
        # torch.nn.CrossEntropyLoss(output.view(-1, ntokens), target)
        ce_loss = F.cross_entropy(output.view(-1, ntokens), target)
        # Modification 2: replace the ce_loss function!!! ---- end ----   -libn

        z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer, current_rho_num)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        mixed_loss.backward()


        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)] [lr: {}]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), args.lr, ce_loss.item()))
            # test_column_sparsity(model)

    if args.verbose:
        writer.add_scalar('Admm_training/Cross_Entropy', ce_loss, current_rho_num*args.epochs+epoch)

        writer.add_scalar('Admm_training/Mixed_loss', mixed_loss.item(), current_rho_num*args.epochs+epoch)

        for k, v in admm_loss.items():
            print("at layer {}, admm loss is {}".format(k, v))

        for k in ADMM.prune_ratios:
            writer.add_scalar('Train_combine_progressive_layer:{} Train/ADMM_Loss'.format(k), admm_loss[k], current_rho_num*args.epochs+epoch)

        return mixed_loss

# ?1:
# Seemingly, there is no difference between: masked_retrain() and combined_masked_retrain(). -libn
# Used for retraining: to update: Weight according to loss(output, target). -libn
def combined_masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch, TEXT, writer):
    if not args.masked_retrain:
        return

    idx_loss_dict = {}

    model.train()
    masks = {}

    # The only difference between masked_retrain() and combined_masked_retrain() ??? -libn
    # get masks from parameters! -libn
    with open("./profile/" + args.config_file + ".yaml", "r") as stream:
        raw_dict = yaml.load(stream)
        prune_ratios = raw_dict['prune_ratios']
    # The only difference between masked_retrain() and combined_masked_retrain() ??? -libn
    
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        _, weight = weight_pruning(args, W, prune_ratios[name])
        W.data = W
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Modification 1: added by libn for transformer!!! ---- start ----   -libn
        data = data.reshape([data.shape[1],data.shape[2]])
        target = target.reshape(target.shape[1])
        # Modification 1: added by libn for transformer!!! ---- end ----    -libn
        optimizer.zero_grad()
        output = model(data)

        # loss = F.cross_entropy(output, target)
        # Modification 2: replace the ce_loss function!!! ---- start ----   -libn
        ntokens = len(TEXT.vocab.stoi)
        # torch.nn.CrossEntropyLoss(output.view(-1, ntokens), target)
        loss = F.cross_entropy(output.view(-1, ntokens), target)
        # Modification 2: replace the ce_loss function!!! ---- end ----   -libn

        loss.backward()

        # added: gradient clipping. -libn
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("({}) cross_entropy loss: {}".format(args.sparsity_type, loss))
            print('re-Train Epoch: {} [{}/{} ({:.0f}%)] [lr: {}]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), current_lr, loss.item()))

        if batch_idx % 10 == 0:
            idx_loss_dict[batch_idx] = loss.item()


    if args.verbose:
        writer.add_scalar('Retrain_combine_progressive/Cross_Entropy', loss.item(), epoch)

        # test_filter_sparsity(model)


        # test_sparsity(args, ADMM, model)
    return idx_loss_dict, loss.item()

# ?1:
# Seemingly, there is no difference between: masked_retrain() and combined_masked_retrain(). -libn
def masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch, TEXT):
    if not args.masked_retrain:
        return

    idx_loss_dict = {}

    model.train()
    masks = {}
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        above_threshold, W = weight_pruning(args, W, ADMM.prune_ratios[name])
        W.data = W
        masks[name] = above_threshold

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("({}) cross_entropy loss: {}".format(args.sparsity_type, loss))
            print('re-Train Epoch: {} [{}/{} ({:.0f}%)] [{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), current_lr, loss.item()))

        if batch_idx % 1 == 0:
            idx_loss_dict[batch_idx] = loss.item()

        # test_sparsity(args, ADMM, model)
    # admm.test_sparsity(args, ADMM, model)
    return idx_loss_dict

