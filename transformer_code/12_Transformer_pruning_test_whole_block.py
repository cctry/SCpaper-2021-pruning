
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import argparse

from torch.utils.tensorboard import SummaryWriter

import testers
import admm_whole_block as admm_transformer

import os

from torch.utils.data import Dataset, DataLoader

import itchat

import platform
pv=int(platform.python_version()[0])        # Check python version
if pv>2:
    import _thread as th
else:
    import thread as th

import os
import time

# 1) Transformer class
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 1) batchify: vector -> [data_size, batch_size] matrix
def batchify(TEXT, data, bsz, device):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

# 2) build sequence based on bptt: [data_size, batch_size] matrix -> [index, data_size, batch_size] matrix] data
def sequence_bptt(data, bptt):
    len_data = data.shape[0]
    inputs = torch.zeros((int(len_data/bptt), bptt, data.shape[1]),dtype=data.dtype) # data.shape[1]:batch_size
    outputs = torch.zeros(int(len_data/bptt), bptt*data.shape[1],dtype=data.dtype)
    
    for i in range(int(len_data/bptt)-1):
        inputs[i,:,:] = data[i:i+bptt]
        outputs[i,:] = data[i+1:i+1+bptt].view(-1)
    return inputs, outputs

class TransformerDataset(Dataset):
    def __init__(self, data_inputs, data_outputs):
        self.len = data_inputs.shape[0]
        # DATASET!
        self.inputs = data_inputs
        self.outputs = data_outputs
        # self.inputs = torch.tensor(datadata_size, input_size)
        # self.outputs = torch.randn(data_size, output_size)
    def __getitem__(self, index):
        return (self.inputs[index].reshape([self.inputs.shape[1],self.inputs.shape[2]]),
        self.outputs[index].reshape(self.outputs.shape[1]))
    def __len__(self):
        return self.len

# def get_batch(bptt, source, i):
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target

# Training using ADMM without masks (loss_sum = loss(output,target)+loss(U,Z)). -libn
def train(args, ADMM, model, device, train_loader, optimizer, epoch, writer):
    model.train()

    ce_loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.cross_entropy(output, target)

        admm_transformer.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm_transformer.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        mixed_loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))

    if args.verbose:
        writer.add_scalar('Train/Cross_Entropy', ce_loss, epoch)
        for k, v in admm_loss.items():
            print("at layer {}, admm loss is {}".format(k, v))

        for k in ADMM.prune_ratios:
            writer.add_scalar('layer:{} Train/ADMM_Loss'.format(k), admm_loss[k], epoch)


def test(args, model, device, test_loader, TEXT):
    model.eval()
    test_loss = 0
    accuracy = 0
    ntokens = len(TEXT.vocab.stoi)
    rng_state = torch.get_rng_state()
    print("rng state is: ", rng_state)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Modification 1: added by libn for transformer!!! ---- start ----   -libn
            data = data.reshape([data.shape[1],data.shape[2]])
            target = target.reshape(target.shape[1])
            # Modification 1: added by libn for transformer!!! ---- end ----    -libn

            # print("data.shape=",data.shape,"target.shape",target.shape)
            output = model(data)

            # Modification 2: replace the ce_loss function!!! ---- start ----   -libn
            # ntokens = len(TEXT.vocab.stoi)
            test_loss += F.cross_entropy(output.view(-1, ntokens), target)
            # Modification 2: replace the ce_loss function!!! ---- end ----   -libn

            output_flat = output.view(-1, ntokens)
            _, indices = output_flat.max(1)
            accuracy = np.mean((accuracy, ((target==indices).sum()/float(len(target))).cpu().detach().numpy()))


    test_loss /= len(test_loader.dataset)
    np.save('./rng_state/rng_state_{:.4f}.npy'.format(accuracy), rng_state)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    # return (100. * correct / len(test_loader.dataset))
    return accuracy

# (1) Irregular pruning:
def prune_const(weight, percent): # remove elements that below the percent percentile. -libn
    weight_np = weight.cpu().detach().numpy()
    pcen = np.percentile(abs(weight_np), percent)
    threshold = abs(weight_np) < pcen
    # threshold = abs(weight_np) < pcen+0.01
    weight_np[threshold] = 0.
    # print('Weight_np: max: %.4f, min: %.4f' %(weight_np.max(),weight_np.min()))
    return torch.nn.Parameter(torch.tensor(weight_np))

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Transformer ADMM Pruning Example')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    # parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=int, default=30, metavar='LR_decay',
                        help='how many every epoch before lr drop (default: 30)')
    # parser.add_argument('--optmzr', type=str, default='sgd', metavar='OPTMZR',
    #                     help='optimizer used (default: sgd)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--save_model', type=str, default="pretrained_mnist.pt",
    #                     help='For Saving the current Model')
    parser.add_argument('--load_model', type=str, default="transformer_model.pt",
                        help='For loading the model')
    parser.add_argument('--masked_retrain', action='store_true', default=False,
                        help='for masked retrain')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='whether to report admm convergence condition')
    parser.add_argument('--admm_transformer', action='store_true', default=False,
                        help="for admm training")
    parser.add_argument('--combine_progressive', action='store_true', default=False,
                        help="for filter pruning after column pruning")
    parser.add_argument('--admm_epoch', type=int, default=9,
                        help="how often we do admm update")
    parser.add_argument('--rho', type=float, default=0.001,
                        help="define rho for ADMM")
    parser.add_argument('--sparsity_type', type=str, default='block_wise',
                        help="define sparsity_type: [irregular,column,filter,pattern,random-pattern]")
    parser.add_argument('--block_size', type=int, default=16,
                        help="block size for block_wise pruning")                   
    parser.add_argument('--block_row_width', type=int, default=16,
                        help="block row width for block_wise pruning") 
    parser.add_argument('--block_col_width', type=int, default=16,
                        help="block column width for block_wise pruning") 
    parser.add_argument('--config_file', type=str, default='config_transformer_2',
                        help="prune config file")
    parser.add_argument('--rho_num', type=int, default=1,
                        help="define how many rohs for ADMM training")
    parser.add_argument('--lr_scheduler', type=str, default='exponential',
                        help='define lr scheduler')
    parser.add_argument('--enable_wechat', action='store_true', default=False,
                        help='enable wechat file transfer')
    args, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    print('block size: ', args.block_size)
    # args, unparsed = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Wechat interaction:  Failed! Login timeout!
    if args.enable_wechat:
        itchat.auto_login(True, enableCmdQR=2)
        # itchat.run() 


    # # # Add: model pruning: the larger percentage, the larger ratio of compression!
    # pruning_ratio = {'self_attn_0':100, 'linear1_0':100, 'linear2_0':100, 'self_attn_1':100, 'linear1_1':100, 'linear2_1':100}
    # print('pruning ratio: self_attn_0: %.2f, linear1_0: %.2f, linear2_0: %.2f, self_attn_1: %.2f, linear1_1: %.2f, linear2_1: %.2f' 
    #             %(pruning_ratio['self_attn_0'],pruning_ratio['linear1_0'],pruning_ratio['linear2_0'],
    #             pruning_ratio['self_attn_1'],pruning_ratio['linear1_1'],pruning_ratio['linear2_1']))

    # writer = SummaryWriter('logs/transformer_pruned_{}_{}_{}_{}_{}_{}'.format(
    #                     pruning_ratio['self_attn_0'],pruning_ratio['linear1_0'],pruning_ratio['linear2_0'],
    #                     pruning_ratio['self_attn_1'],pruning_ratio['linear1_1'],pruning_ratio['linear2_1']))


    import torchtext
    from torchtext.data.utils import get_tokenizer
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print('device: ', device)



    # 2) Dataset: train & test dataset! 
    batch_size = args.batch_size   # 20 for PC, 800 for Server!
    eval_batch_size = 10

    # 2-1) batchify: vector -> [data_size, batch_size] matrix
    train_data = batchify(TEXT, train_txt, batch_size, device)
    val_data = batchify(TEXT, val_txt, eval_batch_size, device)
    test_data = batchify(TEXT, test_txt, eval_batch_size, device)

    # 2-2) build sequence based on bptt: [data_size, batch_size] matrix -> [index, data_size, batch_size] matrix] data
    bptt = 35
    train_data_bptt_inputs, train_data_bptt_outputs = sequence_bptt(train_data, bptt)
    val_data_bptt_inputs, val_data_bptt_outputs = sequence_bptt(val_data, bptt)
    test_data_bptt_inputs, test_data_bptt_outputs = sequence_bptt(test_data, bptt)

    train_loader = DataLoader(dataset=TransformerDataset(train_data_bptt_inputs, train_data_bptt_outputs),
                            shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=TransformerDataset(val_data_bptt_inputs, val_data_bptt_outputs),
                            shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=TransformerDataset(test_data_bptt_inputs, test_data_bptt_outputs),
                            shuffle=True, num_workers=1)


    ######################################################################
    # The model is set up with the hyperparameter below. The vocab size is
    # equal to the length of the vocab object.
    #

    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    emsize = 800 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    # nhead = 2 # the number of heads in the multiheadattention models
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    if args.enable_wechat:
        itchat.send("Let's start working!", toUserName='filehelper')                    

    ### ADMM training ###
    ###    start:     ###

    """====================="""
    """ multi-rho admm train"""
    """====================="""

    initial_rho = args.rho
    if args.admm_transformer:
        writer = SummaryWriter('logs/transformer_model_ADMM_training_rhonum_{}_initial_rho_{}_{}_{}_{}'.format(args.rho_num, initial_rho, args.config_file, args.block_size, args.lr))
        for i in range(args.rho_num):
            current_rho = initial_rho * 10 ** i
            if i == 0:
                model.load_state_dict(torch.load('./models/'+args.load_model))  # admm train need basline model
                print('Pre-train mode: {} loadedl!!!!!!'.format(args.load_model))
                testers.test_irregular_sparsity(model)
                model.cuda()
                save_dir = './models_ADMM'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
            else:
                # continue training based on the previous trained model. -libn
                model.load_state_dict(torch.load(save_dir+'/transformer_model_admm_training_{}_th_rho_{}_{}_{}_{}_{}_{}.pt'.format(i-1, current_rho/10, args.config_file, args.sparsity_type, args.epochs, args.block_size, args.lr)))
                model.cuda()

            accuracy = test(args, model, device, test_loader, TEXT)
            print('Accuracy: %.4f' % accuracy)

            ADMM = admm_transformer.ADMM(model, "./profile/" + args.config_file + ".yaml", rho=current_rho)
            admm_transformer.admm_initialization(args, ADMM, model)  # intialize Z and U variables

            best_accuracy = 0.
            lr = args.lr / 10
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_data), eta_min=4e-08)
            scheduler_exp = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
            for epoch in range(1, args.epochs + 1):
                admm_transformer.admm_adjust_learning_rate(optimizer, epoch, args)
                if args.lr_scheduler == 'default':
                    adjust_learning_rate(optimizer, epoch, args)
                elif args.lr_scheduler == 'cosine':
                    scheduler_cosine.step()
                elif args.lr_scheduler == 'exponential':
                    scheduler_exp.step()

                print("current rho: {}".format(current_rho))
                
                if args.combine_progressive: # ADMM training with masks to constraint weight gradients!
                    mixed_loss = admm_transformer.admm_masked_train(args, ADMM, model, device, train_loader, optimizer, epoch, TEXT, writer, current_rho_num=i)
                else:                        # ADMM training without masks to constraint weight gradients!
                    train(args, ADMM, model, device, train_loader, optimizer, epoch, writer)

                accuracy = test(args, model, device, test_loader, TEXT)
                print('Accuracy: %.4f' % accuracy)
                best_accuracy = max(best_accuracy, accuracy)

                if args.enable_wechat:
                    # if (epoch%10==0 & epoch!=0):
                    itchat.send('Current training configuration: \n lr_initial = {}, rho = {}, config_file = {}, sparsity_type = {}, epochs = {}, block_size = {}'.format(args.lr, current_rho, args.config_file, args.sparsity_type, args.epochs, args.block_size), toUserName='filehelper')
                    itchat.send('Current training result: \n Epoch = {}, Current lr = {}, Best accuracy = {}, mixed_loss = {}'.format(i*args.epochs+epoch, optimizer.param_groups[0]['lr'], best_accuracy, mixed_loss), toUserName='filehelper')

                print('Current training result: \n Epoch = {}, Current lr = {}, Best accuracy = {}, mixed_loss = {}'.format(i*args.epochs+epoch, optimizer.param_groups[0]['lr'], best_accuracy, mixed_loss))

                writer.add_scalar('Admm_training/Accuracy', accuracy, int(i*args.epochs+epoch))
                writer.add_scalar('Admm_training/current_lr', optimizer.param_groups[0]['lr'], int(i*args.epochs+epoch))

            print("Saving ADMM model...")
            print('args.block_size: ', args.block_size)
            torch.save(model.state_dict(), save_dir+'/transformer_model_admm_training_{}_th_rho_{}_{}_{}_{}_{}_{}.pt'.format(i, current_rho, args.config_file, args.sparsity_type, args.epochs, args.block_size, args.lr))

            print('ADMM-trained model saved !!!!!!')

            print('After Admm-training, test the weight sparsity of the model: ')
            print('Current training configuration: \n lr_initial = {}, rho = {}, config_file = {}, sparsity_type = {}, epochs = {}, block_size = {}'.format(args.lr, current_rho, args.config_file, args.sparsity_type, args.epochs, args.block_size))          
            testers.test_irregular_sparsity(model)

        print("!!!!!! Accuracy after admm training:")
        accuracy = test(args, model, device, test_loader, TEXT)
        print('Accuracy: %.4f' % accuracy)




    """========================"""
    """END multi-rho admm train"""
    """========================"""


    """=============="""
    """masked retrain"""
    """=============="""

    if args.masked_retrain:
        writer = SummaryWriter('logs/transformer_model_ADMM_retraining_{}__{}'.format(args.load_model, args.lr))
        # load admm trained model
        print("\n>_ Loading file...")
        model.load_state_dict(torch.load("./models_ADMM/"+args.load_model))
        model.cuda()

        print("!!!!!! Accuracy before hard pruning:")
        accuracy = test(args, model, device, test_loader, TEXT)
        print('Accuracy: %.4f' % accuracy)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        ADMM = admm_transformer.ADMM(model, file_name="./profile/" + args.config_file + ".yaml", rho=initial_rho)
        admm_transformer.hard_prune(args, ADMM, model)

        print("!!!!!! Accuracy after hard pruning:")
        accuracy = test(args, model, device, test_loader, TEXT)
        print('Accuracy: %.4f' % accuracy)

        admm_transformer.test_sparsity(args, ADMM, model)
        best_prec1 = [0]
        epoch_loss_dict = {}
        testAcc = []
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_data), eta_min=4e-08)
        scheduler_exp = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        for epoch in range(1, args.epochs + 1):
            if epoch == 1:
                save_dir = './models_ADMM_pruned'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

            if args.lr_scheduler == 'cosine':
                scheduler_cosine.step()
            elif args.lr_scheduler == 'exponential':
                scheduler_exp.step()

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.lr * (0.5 ** (epoch // args.lr_decay))
            if args.combine_progressive:
                idx_loss_dict, loss_retraining = admm_transformer.combined_masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch, TEXT, writer)
            else:
                idx_loss_dict = admm_transformer.masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch, TEXT)
            prec1 = test(args, model, device, test_loader, TEXT)
            prec1 = round(prec1, 3) # save the accuracy using 3 decimal places to fix saving model error! -libn
            print('Accuracy: %.4f' % prec1)
            writer.add_scalar('Retrain_combine_progressive/Accuracy', prec1, epoch)

            if prec1 > max(best_prec1):
                # Error happens when previous accuracy = 0.75401, current accuracy = 0.75402; then, model_0.754 is deleted! => model_0.754 not found error! -libn
                print("\n>_ Got better accuracy, saving model with accuracy {:.4f}% now...\n".format(prec1))
                torch.save(model.state_dict(),save_dir+"/transformer_retrained_acc_{:.4f}_{}__{}.pt".format(prec1, args.load_model, args.lr))                               
                print("\n>_ Deleting previous model file with accuracy {:.4f}% now...\n".format(max(best_prec1)))
                if len(best_prec1) > 1:
                    os.remove(save_dir+"/transformer_retrained_acc_{:.4f}_{}__{}.pt".format(max(best_prec1), args.load_model, args.lr)) 

            epoch_loss_dict[epoch] = idx_loss_dict
            testAcc.append(prec1)

            best_prec1.append(prec1)
            print('best_accuracy： ', best_prec1)

            if args.enable_wechat:
                # if (epoch%10==0 & epoch !=0):
                itchat.send('Current retraining configuration: \n model = {}; lr = {}'.format(args.load_model, args.lr), toUserName='filehelper')
                itchat.send('Current retraining result: \n Epoch = {}, Current lr = {}, Best accuracy = {}, ce_loss = {}'.format(epoch, optimizer.param_groups[0]['lr'], best_prec1, loss_retraining), toUserName='filehelper')
            
            writer.add_scalar('Retrain_combine_progressive/current_lr', optimizer.param_groups[0]['lr'], epoch)



        print("!!!!!! Accuracy after retraining")
        accuracy = test(args, model, device, test_loader, TEXT)
        print('Accuracy: %.4f' % accuracy)
        print('ADMM-pruned & trained model!!!!!!')

        print('After Retraining, test the weight sparsity of the model: ')
        print('Current retraining configuration: \n model = {}; lr = {}'.format(args.load_model, args.lr))
        testers.test_irregular_sparsity(model)
        for update_name, update_weight in model.named_parameters():
            if ('in_proj_weight' in update_name) or ('out_proj.weight' in update_name) \
                    or ('linear1.weight' in update_name) or ('linear2.weight' in update_name) \
                    or ('encoder.weight' in update_name) or ('decoder.weight' in update_name):
                plot_heatmap(update_name, update_weight, "./figures/")
        compression_rate = testers.test_irregular_sparsity(model)
        writer.add_scalar('Retrain_combine_progressive/compression_rate', compression_rate, epoch)  # Save the calculated compression rate only once! -libn
        # admm.test_sparsity(args, ADMM, model)

        print("Best Acc: {:.4f}".format(max(best_prec1)))
        save_dir = './plotable'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir+"/plotable_{}.npy".format(args.sparsity_type), epoch_loss_dict)
        np.save(save_dir+"/testAcc_{}.npy".format(args.sparsity_type), testAcc)


    """=============="""
    """masked retrain"""
    """=============="""



    ###     end.      ###
    ### ADMM training ###
    writer.close()


import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(name, weight, file_path):
    weight = weight.cpu().detach().numpy()
    weight2d = weight.reshape(weight.shape[0], -1)
    im = plt.matshow(np.abs(weight2d), cmap=plt.cm.binary, aspect='equal')
    plt.colorbar(im)
    plt.title(name)
    plt.savefig(file_path + name + '.jpg', dpi=800)
    plt.show()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()
