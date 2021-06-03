
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import argparse

from torch.utils.tensorboard import SummaryWriter


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

def batchify(TEXT, data, batch_size, device):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch(bptt, source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

import time
def train(args, model, TEXT, train_data, bptt, optimizer, criterion, pruning_ratio, device, epoch, scheduler):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(bptt, train_data, i) # data.shape = [bptt,batch_size], targets.shape = [(bptt*batch_size),1]
        optimizer.zero_grad()

        if args.irregular_pruning:
            # Add: model pruning: the larger percentage, the larger ratio of compression!
            temp = model.transformer_encoder.layers._modules['0'].self_attn.out_proj.weight
            model.transformer_encoder.layers._modules['0'].self_attn.out_proj.weight = prune_const(temp, pruning_ratio['self_attn_0'])
            temp = model.transformer_encoder.layers._modules['0'].linear1.weight
            model.transformer_encoder.layers._modules['0'].linear1.weight = prune_const(temp, pruning_ratio['linear1_0'])
            temp = model.transformer_encoder.layers._modules['0'].linear2.weight
            model.transformer_encoder.layers._modules['0'].linear2.weight = prune_const(temp, pruning_ratio['linear2_0'])
            temp = model.transformer_encoder.layers._modules['1'].self_attn.out_proj.weight
            model.transformer_encoder.layers._modules['1'].self_attn.out_proj.weight = prune_const(temp, pruning_ratio['self_attn_1'])
            temp = model.transformer_encoder.layers._modules['1'].linear1.weight
            model.transformer_encoder.layers._modules['1'].linear1.weight = prune_const(temp, pruning_ratio['linear1_1'])
            temp = model.transformer_encoder.layers._modules['1'].linear2.weight
            model.transformer_encoder.layers._modules['1'].linear2.weight = prune_const(temp, pruning_ratio['linear2_1'])
            model.to(device)

        output = model(data)

        # print('Output:')
        # print(output)

        # print('Target:')
        # print(targets)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()




        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    
def evaluate(eval_model, data_source, TEXT, bptt, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(bptt, data_source, i) # data.shape=[bptt,eval_batch_size], targets.shape=[(bptt*targets.shape),1]
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()

            _, indices = output_flat.max(1)
            accuracy = (targets==indices).sum()/float(len(targets))
    return total_loss / (len(data_source) - 1), accuracy



# (1) Irregular pruning:
def prune_const(weight, percent): # remove elements that below the percent percentile. -libn
    weight_np = weight.cpu().detach().numpy()
    pcen = np.percentile(abs(weight_np), percent)
    threshold = abs(weight_np) < pcen
    # threshold = abs(weight_np) < pcen+0.01
    weight_np[threshold] = 0.
    # print('Weight_np: max: %.3f, min: %.3f' %(weight_np.max(),weight_np.min()))
    return torch.nn.Parameter(torch.tensor(weight_np))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Transformer Pruning Example')
    parser.add_argument('--train', action='store_true', default=True,
                        help='start training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='start testing')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save derived model')
    parser.add_argument('--irregular_pruning', action='store_true', default=False,
                        help='start pruning using irregular pruning method')

    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    # parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=3.0, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--lr_decay', type=int, default=30, metavar='LR_decay',
    #                     help='how many every epoch before lr drop (default: 30)')
    # parser.add_argument('--optmzr', type=str, default='sgd', metavar='OPTMZR',
    #                     help='optimizer used (default: sgd)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log_interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')

    parser.add_argument('--load_model', type=str, default="transformer_model.pt",
                        help='For loading the model')
    # parser.add_argument('--masked_retrain', action='store_true', default=False,
    #                     help='for masked retrain')
    # parser.add_argument('--verbose', action='store_true', default=False,
    #                     help='whether to report admm convergence condition')
    # parser.add_argument('--admm', action='store_true', default=False,
    #                     help="for admm training")
    # parser.add_argument('--combine_progressive', action='store_true', default=False,
    #                     help="for filter pruning after column pruning")
    # parser.add_argument('--admm_epoch', type=int, default=9,
    #                     help="how often we do admm update")
    # parser.add_argument('--rho', type=float, default=0.001,
    #                     help="define rho for ADMM")
    # parser.add_argument('--sparsity_type', type=str, default='pattern',
    #                     help="define sparsity_type: [irregular,column,filter,pattern,random-pattern]")
    # parser.add_argument('--config_file', type=str, default='config',
    #                     help="prune config file")
    # parser.add_argument('--rho_num', type=int, default=1,
    #                     help="define how many rohs for ADMM training")
    # parser.add_argument('--lr_scheduler', type=str, default='cosine',
    #                     help='define lr scheduler')


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.irregular_pruning:
        # Add: model pruning: the larger percentage, the larger ratio of compression!
        pruning_ratio = {'self_attn_0':100, 'linear1_0':100, 'linear2_0':100, 'self_attn_1':100, 'linear1_1':100, 'linear2_1':100}
        print('pruning ratio: self_attn_0: %.2f, linear1_0: %.2f, linear2_0: %.2f, self_attn_1: %.2f, linear1_1: %.2f, linear2_1: %.2f' 
                    %(pruning_ratio['self_attn_0'],pruning_ratio['linear1_0'],pruning_ratio['linear2_0'],
                    pruning_ratio['self_attn_1'],pruning_ratio['linear1_1'],pruning_ratio['linear2_1']))

        writer = SummaryWriter('logs/transformer_pruned_{}_{}_{}_{}_{}_{}'.format(
                            pruning_ratio['self_attn_0'],pruning_ratio['linear1_0'],pruning_ratio['linear2_0'],
                            pruning_ratio['self_attn_1'],pruning_ratio['linear1_1'],pruning_ratio['linear2_1']))
    else:
        pruning_ratio = 0
        writer = SummaryWriter('logs/transformer_original_model_lr_{}_{}'.format(args.lr, args.epochs))


    import torchtext
    from torchtext.data.utils import get_tokenizer
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)     
    # len(train_txt.examples[0].text) = 2086708
    # len(val_txt.examples[0].text) = 218177
    # len(test_txt.examples[0].text) = 246217
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)



    # 2) Dataset: train & test dataset! 
    batch_size = args.batch_size   # 20 for PC, 800 for Server!
    eval_batch_size = 10
    train_data = batchify(TEXT, train_txt, batch_size, device)
    val_data = batchify(TEXT, val_txt, eval_batch_size, device)
    test_data = batchify(TEXT, test_txt, eval_batch_size, device)


    # sequence length
    bptt = 35



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


    criterion = nn.CrossEntropyLoss()
    lr = args.lr # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    ######################################################################
    # Loop over epochs. Save the model if the validation loss is the best
    # we've seen so far. Adjust the learning rate after each epoch.

    best_val_loss = float("inf")
    epochs = args.epochs # The number of epochs
    best_model = None

    if args.train:
        print('Training:')
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(args, model, TEXT, train_data, bptt, optimizer, criterion, pruning_ratio, device, epoch, scheduler)
            val_loss, accuracy = evaluate(model, val_data, TEXT, bptt, criterion)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss), accuracy))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

            scheduler.step()

            # write to tensorboard
            writer.add_scalar('Original model_Loss/train', val_loss, epoch)
            writer.add_scalar('Original model_Accuracy/train', accuracy, epoch)
            writer.close()


        ######################################################################
        # Evaluate the model with the test dataset
        # -------------------------------------
        #
        # Apply the best model to check the result with the test dataset.

        test_loss, test_accuracy = evaluate(best_model, test_data, TEXT, bptt, criterion)



        # Save trained model:
        if args.save_model:
            save_dir = './models/'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), save_dir+'transformer_model_lr_{}_{}.pt'.format(args.lr, args.epochs))
            print('!!!!!!!!! model: transformer_model_lr_{}_{}.pt saved!'.format(args.lr, args.epochs))


        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)

    if args.test:
        print('Testing:')

        # Load the model:
        model.load_state_dict(torch.load('./models/'+args.load_model))  # admm train need basline model
        test_loss, test_accuracy = evaluate(model, test_data, TEXT, bptt, criterion)

        print('=' * 89)
        print(args.load_model + '| test loss {:5.4f} | test ppl {:8.4f} | accuracy {:8.4f}'.format(
            test_loss, math.exp(test_loss), test_accuracy))
        print('=' * 89)
            



if __name__ == '__main__':
    main()
