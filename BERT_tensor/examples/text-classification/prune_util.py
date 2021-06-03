from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from testers import *

# row pruning for each entire block of rows: -libn
def block_rows_pruning(args, block_weight_np, prune_threshold):

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
    percent = prune_threshold * 100
    for gp in range(groups_shape[0]):
        # for each small block (weight_groups[gp]): -libn
        # Step 2: prune each block using column pruning:
        # group_mask[gp, :, :], weight_groups[gp, :, :] = rows_pruning(args, weight_groups[gp], prune_threshold)


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
        # above_threshold_remain, weight_remain = block_rows_pruning(args, weight_remain, prune_threshold)

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



def random_pruning(args, weight, prune_threshold):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_threshold), replace=False)
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


def L1_pruning(args, weight, prune_threshold):
    """
    projected gradient descent for comparison
    """
    percent = prune_threshold * 100
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


def weight_pruning(args, weight, prune_threshold):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_threshold (float between 0-1): target sparsity of weights
    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero
    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    if args.sparsity_type == "irregular":
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, prune_threshold*100)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        #return above_threshold.to(args.device), torch.from_numpy(weight).to(args.device)
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, prune_threshold*100)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "channel"):
        shape = weight.shape
        print("channel pruning...", weight.shape)
        weight3d = weight.reshape(shape[0], shape[1], -1)
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2))
        percentile = np.percentile(channel_l2_norm, prune_threshold*100)
        under_threshold = channel_l2_norm <= percentile
        above_threshold = channel_l2_norm > percentile
        weight3d[:,under_threshold,:] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32)
        for i in range(weight3d.shape[1]):
            expand_above_threshold[:, i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, prune_threshold*100)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        # weight2d[weight2d < 1e-40] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "block_filter"): # -libn

        shape = weight.shape
        conv = weight.reshape(shape[0], -1)
        shape2d = conv.shape
        # print('weight.shape', conv.shape)

        block_row_division = args.block_row_division
        block_row_width = args.block_row_width
        print('conv.shape[1]',conv.shape[1])
        print('block_row_width',block_row_width)
        print('block_row_division',block_row_division)

        if conv.shape[1]%block_row_division != 0 :
            print("the layer size is not divisible by block_row_division",conv.shape[1], block_row_division)
            # raise SyntaxError("block_size error")
        block_number = int(conv.shape[1]/block_row_division)
        print('after block_row_division int',block_number)
        convfrag = torch.chunk(torch.tensor(conv), block_number, dim=1)
        print('convfrag',len(convfrag))

        mat = None
        for k in range(len(convfrag)):
            if mat is None:
                mat = convfrag[k]
            else:
                mat = torch.cat((mat, convfrag[k]), 0)
        # calculate all the norms of each block (block.shape = [1, conv.shape[1]/block_row_division])
        row_norms = torch.norm(mat.data, dim=1)

        if args.prune_ratio_config != None:
            #print('!!! prune_ratio = ', prune_threshold)
            prune_threshold = np.percentile(row_norms, prune_threshold*100)  # get a value for this percentitle
            
        under_threshold = row_norms < prune_threshold
        above_threshold = row_norms > prune_threshold
        print('under_threshold, above_threshold',len(under_threshold),len(above_threshold))
       # print('!!! prune_threshold = ', prune_threshold)
        # above_threshold = above_threshold.astype(
        #     np.float32)  # has to convert bool to float32 for numpy-tensor conversion

        # to construct the under_threshold matrix (under_threshold.shape = weight.shape)
        print('before under block_number',block_number)
        print('int(conv.shape[1]/block_number)',int(conv.shape[1]/block_number))
        print('under_threshold.reshape((len(under_threshold),1))',under_threshold.reshape((len(under_threshold),1)).shape)
        under_threshold_matrix = torch.tensor(np.tile(under_threshold.reshape((len(under_threshold),1)), int(conv.shape[1]/block_number)))
        #print('under_threshold_matrix',under_threshold_matrix)
        print('under_threshold_matrix.shape',under_threshold_matrix.shape)
        print('int(under_threshold_matrix.shape[0]/conv.shape[0])',int(under_threshold_matrix.shape[0]/conv.shape[0]))
        convfrag = torch.chunk(under_threshold_matrix, int(under_threshold_matrix.shape[0]/conv.shape[0]), dim=0)
        print('convfrag.shape after under',len(convfrag))
        mat = None
        for m in range(len(convfrag)):
            #print('convfrag[m] after under',convfrag[m].shape[0])
            if mat is None:
                mat = convfrag[m]
            else:
                mat = torch.cat((mat, convfrag[m]), 1)
        under_threshold_matrix = mat
        
        print('under_threshold_matrix',len(under_threshold_matrix))

        conv[under_threshold_matrix] = 0

        above_threshold = ~under_threshold_matrix

        conv = conv.reshape(weight.shape)
        above_threshold = above_threshold.reshape(weight.shape)

        return above_threshold.to(args.device), torch.from_numpy(weight).to(args.device)
    elif (args.sparsity_type == "block_column"): # -libn

        shape = weight.shape
        conv = weight.reshape(shape[0], -1)
        print('weight.shape', conv.shape)

        block_row_division = args.block_row_division
        block_row_width = args.block_row_width
        print('block_row_width',block_row_width)
        print('block_row_division',block_row_division)
        #if conv.shape[0]==30522:
        #   block_row_division=30522


        block_number = int(conv.shape[0]/block_row_width)
        block_number_noint = conv.shape[0]/block_row_width
            #print('block_row_division int',block_row_division)

        print('block_row_width int',block_number)
        #if block_number==170:
        diff_conv=0
        if block_number!=block_number_noint:
            block_number+=1
            diff_conv=block_number*block_row_width-conv.shape[0]
            print('diff_conv', diff_conv)
            arrayzero=torch.zeros(diff_conv,conv.shape[1])
            print('arrayzero',arrayzero.shape)
            conv=torch.cat((torch.tensor(conv),arrayzero),0)
            print('conv+arrayzero',conv.shape)
            #conv=conv[:-diff]
            #print('conv[:-diff]',conv.shape)
            convfrag = torch.chunk(torch.tensor(conv), block_number, dim=0)
        #elif block_number==0:
             #block_number=1
             #convfrag = torch.chunk(torch.tensor(conv), 1, dim=0)         
        else:   
             convfrag = torch.chunk(torch.tensor(conv), block_number, dim=0)
        print('len(convfrag)',len(convfrag))
        mat = None
        for k in range(len(convfrag)):
            #if conv.shape[0]==512:
            print('convfrag[k]',convfrag[k].shape)
            if mat is None:
                mat = convfrag[k]
            else:
                if convfrag[k].shape[0]!=block_row_width:
                    print('**********convfrag[k]',convfrag[k].shape)
                    diff=convfrag[k].shape[0]-block_row_width
                    print('diff', diff)
                    arrayzero=torch.zeros(diff,convfrag[k].shape[1])
                    print('arrayzero',arrayzero.shape)
                    combine=torch.cat((convfrag[k],arrayzero),0)
                    print('convfrag[k]',combine.shape)
                    mat = torch.cat((mat, combine), 1)

                #if convfrag[k].shape[0]==2:
                    #print('**********convfrag[k]',convfrag[k].shape)
                    #arrayzero=torch.zeros(1,768)
                    #print('arrayzero',arrayzero.shape)
                    #combine=torch.cat((convfrag[k],arrayzero),0)
                    #print('convfrag[k]',combine.shape)
                    #mat = torch.cat((mat, combine), 1)
                else:
                    mat = torch.cat((mat, convfrag[k]), 1)
        # calculate all the norms of each block (block.shape = [1, conv.shape[1]/block_row_division])
        row_norms = torch.norm(mat.data, dim=0)
        print('row_norms',len(row_norms))
        if args.prune_ratio_config != None:
            #print('!!! prune_ratio = ', prune_threshold)
            prune_threshold = np.percentile(row_norms, prune_threshold*100)  # get a value for this percentitle
            
        under_threshold = row_norms < prune_threshold
        above_threshold = row_norms > prune_threshold
        print('under_threshold, above_threshold',len(under_threshold),len(above_threshold))
        #print('!!! prune_threshold = ', prune_threshold)
        # above_threshold = above_threshold.astype(
        #     np.float32)  # has to convert bool to float32 for numpy-tensor conversion

        # to construct the under_threshold matrix (under_threshold.shape = weight.shape)
        print('before under block_number',block_number)
        print('int(conv.shape[0]/block_number)',int(conv.shape[0]/block_number))
        print('under_threshold.reshape((1,len(under_threshold)))',under_threshold.reshape((1,len(under_threshold))).shape)
        under_threshold_matrix = torch.tensor(np.tile(under_threshold.reshape((1,len(under_threshold))), (int(conv.shape[0]/block_number),1)))
        #under_threshold_matrix = torch.tensor(np.tile(int(conv.shape[0]/block_row_width),under_threshold.reshape((len(under_threshold),0))))
        print('under_threshold_matrix.shape',under_threshold_matrix.shape)
        print('int(under_threshold_matrix.shape[1]/conv.shape[1])',int(under_threshold_matrix.shape[1]/conv.shape[1]))
        convfrag = torch.chunk(under_threshold_matrix, int(under_threshold_matrix.shape[1]/conv.shape[1]), dim=1)
        print('convfrag after under',len(convfrag))
        mat = None
        for m in range(len(convfrag)):
            #print('convfrag[m] after under',convfrag[m].shape)
            if mat is None:
                mat = convfrag[m]
            else:
                mat = torch.cat((mat, convfrag[m]), 0)
        #if conv.shape[0]==512:
        #    print('mat before',len(mat))
        #    mat=mat[:-1]
        #    print('mat after',len(mat))
        if diff_conv!=0:
            mat=mat[:-diff_conv]
            conv=conv[:-diff_conv]
            print('conv[:-diff_conv]',conv.shape)
        under_threshold_matrix = mat
        print('under_threshold_matrix',len(under_threshold_matrix))

        conv[under_threshold_matrix] = 0

        above_threshold = ~under_threshold_matrix

        conv = conv.reshape(weight.shape)
        above_threshold = above_threshold.reshape(weight.shape)

        return above_threshold.to(args.device), torch.from_numpy(weight).to(args.device)
    elif (sparsity_type == "whole_block_padding_balanced"):
        shape = weight.shape
        conv = weight.reshape(shape[0], -1)  # type(conv) = np.array
        shape = conv.shape
        # block_row_width = args.block_row_width
        # block_col_width = args.block_col_width
        block_row_width = 24  # block_sizes[0]
        block_col_width = 1  # block_sizes[1]
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
        for i in range(block_row_division):
            temp = conv[i * block_row_width: (i + 1) * block_row_width, :]
            convfrag = torch.chunk(torch.tensor(temp), block_col_division, dim=1)
            convfrag = torch.cat(convfrag, 0)
            block_row_norms = torch.norm(convfrag, dim=1)
            block_norms = torch.chunk(block_row_norms.reshape(1, len(block_row_norms)), block_col_division, dim=1)
            block_norms = torch.cat(block_norms, 0)
            block_norms = torch.norm(block_norms, dim=1)
            if args.prune_ratio_config != None:
                percentile = np.percentile(block_norms, prune_threshold * 100)
            above_threshold = block_norms > percentile
            whole_above_threshold[i * block_col_division:i * block_col_division + block_col_division] = above_threshold[:]
        for kk in range(len(whole_above_threshold)):
            row_start = int(np.floor(kk / block_col_division) * block_row_width)
            col_start = int((kk % block_col_division) * block_col_width)
            mask[row_start:row_start + block_row_width, col_start:col_start + block_col_width] = whole_above_threshold[kk].numpy() * np.ones((block_row_width, block_col_width)).astype('int')
        conv *= mask
        weight = conv[:shape[0], :shape[1]].reshape(weight.shape)
        # print(weight)
        mask = mask[:shape[0], :shape[1]].reshape(weight.shape)
        return torch.from_numpy(mask.reshape(weight.shape)).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "whole_block_padding"):  # -libn
        shape = weight.shape
        conv = weight.reshape(shape[0], -1)
        # Square blocks:
        # block_row_width = args.block_row_width
        # block_col_width = args.block_col_width
        block_row_width = 16
        block_col_width = 16
        print("Block size: %d, %d" % (block_row_width, block_col_width))

        if conv.shape[1] % block_col_width != 0 or conv.shape[0] % block_row_width != 0:
            print("the layer size is not divisible by block_col_width:", conv.shape[0], conv.shape[1], block_row_width,
                  block_col_width)
            padding_height = (block_row_width - conv.shape[0] % block_row_width) % (block_row_width)
            padding_width = (block_col_width - conv.shape[1] % block_col_width) % (block_col_width)
            print("padding: height: %d; width: %d" % (padding_height, padding_width))
            print("data type of conv: ", conv.dtype)
            conv = np.concatenate((conv, np.zeros((padding_height, conv.shape[1])).astype(conv.dtype)), axis=0)
            conv = np.concatenate((conv, np.zeros((conv.shape[0], padding_width)).astype(conv.dtype)), axis=1)
        block_col_division = int(conv.shape[1] / block_col_width)
        convfrag = torch.chunk(torch.tensor(conv), block_col_division, dim=1)
        convfrag = torch.cat(convfrag, 0)
        block_row_norms = torch.norm(convfrag, dim=1)
        block_row_division = int(len(block_row_norms) / block_row_width)
        block_norms = torch.chunk(block_row_norms.reshape(1, len(block_row_norms)), block_row_division, dim=1)
        block_norms = torch.cat(block_norms, 0)
        block_norms = torch.norm(block_norms, dim=1)
        print('!!! prune_ratio = ', prune_threshold)
        prune_threshold = np.percentile(block_norms, prune_threshold * 100)  # get a value for this percentitle
        above_threshold = block_norms > prune_threshold
        above_threshold_chunked_matrix = np.zeros(
            torch.cat(torch.chunk(torch.tensor(conv), block_col_division, dim=1), 0).shape).astype('int')
        for kk in range(len(above_threshold)):
            above_threshold_chunked_matrix[kk * block_row_width:(kk + 1) * block_row_width, :] = above_threshold[
                                                                                                     kk] * np.ones(
                (block_row_width, block_col_width)).astype('int')
        above_threshold_tmp = torch.chunk(torch.tensor(above_threshold_chunked_matrix), block_col_division, dim=0)
        above_threshold_matrix = torch.cat(above_threshold_tmp, 1).numpy()
        conv *= above_threshold_matrix
        above_threshold_matrix = above_threshold_matrix[:shape[0], :shape[1]].reshape(weight.shape)
        weight = conv[:shape[0], :shape[1]].reshape(weight.shape)
        return torch.from_numpy(above_threshold_matrix.reshape(weight.shape)).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")



def hard_prune(args, prune_thresholds, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda
    """

    print("hard pruning")
    for (name, W) in model.named_parameters():
        if name not in prune_thresholds:  # ignore layers that do not have rho
            continue

       # print('!!! Hard prune:', name)
        cuda_pruned_weights = None
        if option == None:
            _, cuda_pruned_weights = weight_pruning(args, W, prune_thresholds[name])  # get sparse model in cuda
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable

        



class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
