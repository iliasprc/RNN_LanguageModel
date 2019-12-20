import torch
import numpy as np
import os
import torch.nn as nn
import time
from torch.utils import data
from utils import *
from dataloader import VideoDataset
from RnnLM import RNNModel
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils import *
import json
from model_utils import *
from calc_wer import calc_wer
from wer_ctc import wer_calc
from jiwer import wer
import argparse
from alignments_novel import get_alignments

best_wer=10000
best_val_loss=2222

train_prefix="train"
test_prefix="dev"
test_filepath="/home/hatzis/Desktop/teo/ctc_last2/files/dev_phoenixv1.csv"

train_filepath="/home/hatzis/Desktop/teo/ctc_last2/files/train_phoenixv1.csv"
ngrams = [2, 3, 4]

def n_gram_batchify(sos, n_gram, device, sequence):

    inputs = torch.tensor([[sos]]).repeat(1, n_gram-1)
    inputs = inputs.to(device)
    inputs = torch.cat( (inputs, sequence[:, :-1] ), 1)
    inputs = inputs.unfold(-1, n_gram-1, 1)

    return inputs.squeeze(0)

def train(args, model, device, train_loader, optimizer, epoch, timestamp, ngrams, id2w):
    model.train()
    count = 0
    train_loss = 0
    total = 0
    sos = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx,  (data) in enumerate(train_loader):

        loss = 0
        data = data.long().to(device)

        inputs = [n_gram_batchify(sos, n_gram, device, data) for n_gram in ngrams]

        for input in inputs:
            output = model(input)
            loss += criterion(output, data)

        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        total += len(data)

        if(batch_idx%args.log_interval==1):

            print('Train Epoch: {} [{}/{}] \t Loss: {:.6f}  '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), train_loss / (batch_idx + 1)))

    cpkt_fol_name='/home/hatzis/Desktop/stim_ctc/multigram/checkpoints/test_day_'+timestamp

    if not os.path.exists(cpkt_fol_name):
        print("Checkpoint Directory does not exist! Making directory {}".format(cpkt_fol_name))
        os.mkdir(cpkt_fol_name)
    logger(cpkt_fol_name+'/training.txt',[str(epoch), str(float(train_loss / (batch_idx + 1))) ])

def validate(args, model, device, test_loader, optimizer, epoch, timestamp, ngrams, id2w):
    model.eval()
    eval_loss = 0
    # correct = 0

    sentences = []
    criterion = nn.CTCLoss(blank=0, reduction='mean')

    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            data = data.long().to(device)
            inputs = [n_gram_batchify(sos, n_gram, device, data) for n_gram in ngrams]

            for input in inputs:
                output = model(input)
                loss += criterion(output, data)

            eval_loss += loss.item()  # sum up batch loss

            probs=nn.functional.softmax(output, dim=-1)
            pred = probs.argmax(dim=-1, keepdim=True).squeeze().cpu().numpy()

            ref = ''
            refs = target.squeeze().cpu().numpy()

            for i in range(target.size(1)):
                ref+=id2w[refs[i]]+' '

            s = greedy_decode(id2w, pred, output.size(0), ' ')

            sentences.append(s)

    #model_out_path = model.get_name() +'no_pad' +'_loss_' +str(float(eval_loss / len(test_loader.dataset)))+'_epoch_'+str(epoch) + ".pth"
    cpkt_fol_name='/home/hatzis/Desktop/stim_ctc/multigram/checkpoints/test_day_'+timestamp
    if not os.path.exists(cpkt_fol_name):
        print("Checkpoint Directory does not exist! Making directory {}".format(cpkt_fol_name))
        os.mkdir(cpkt_fol_name)
    pred_name= cpkt_fol_name+'/subunetseval_greedy_predictions_epoch'+str(epoch)+'loss_'+str(float(eval_loss / len(test_loader.dataset)))+'_'+timestamp+'_.csv'
    write_csv(sentences,pred_name)
    wer=calc_wer("/home/hatzis/Desktop/teo/ctc_last2/files/dev_phoenixv1.csv",pred_name)
    val_loss=eval_loss / len(test_loader.dataset)
    print('Evaluation : Average loss: {:.4f} Word error rate {}%'.format(
        eval_loss / len(test_loader.dataset),wer))

    global best_wer

    for_checkpoint={'epoch':epoch,
                    'model_dict':model.state_dict(),
                    'optimizer_dict':optimizer.state_dict(),
                    'validation_loss':str(val_loss),
                    'word error rate':wer
                    }
    is_best=wer<best_wer
    if(is_best):
        print("BEST WER")
        best_wer=wer
        save_checkpoint(for_checkpoint, is_best,cpkt_fol_name,'best_wer'+str(wer))
    else:
        save_checkpoint(for_checkpoint, is_best,cpkt_fol_name,'last')



    with open(cpkt_fol_name+'/params_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger(cpkt_fol_name+'/validation.txt', [str(epoch), str(float(eval_loss / len(test_loader.dataset))), str(wer)])

def main():


    timestamp=str(time.asctime( time.localtime(time.time())))

    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')


    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    use_cuda = False
    # use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Training settings

    print("DEVICE ",device)

    max_epochs = 100

    N_CL = 47
    dim = (224,224)

    seq_length = 300
    n_channels = 3

    test_params = {'batch_size': 1 ,
                   'shuffle': False,
                   'num_workers': 2}

    train_params = {'batch_size': 1,
                    'shuffle': True,
                    'num_workers': 4}


    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--modality', type=str, default='full_image', metavar='rc',
                        help='hands or full image')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=max_epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataloader', type=str, default='rand_crop', metavar='rc',
                        help='data augmentation')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--seq-length',type=int,default=seq_length,metavar='num',help='squence length')
    parser.add_argument('--hidden_size',type=int,default=1024,metavar='num',help='lstm units')
    parser.add_argument('--optim',type=int,default=1,metavar='optim number',help='optimizer sgd or adam')
    parser.add_argument('--n_layers',type=int,default=2,metavar='num',help='hidden size')
    parser.add_argument('--dropout',type=float,default=0.7,metavar='num',help='hidden size')
    parser.add_argument('--bidirectional',action='store_true',default=True,help='hidden size')

    args = parser.parse_args()

    # Helper: Save the model.

    train_partition,train_labels=load_csv_file(train_filepath)
    test_partition,test_labels=load_csv_file(test_filepath)
    classes=count_classes(train_labels)
    id2w=list(range(len(classes)))
    id2w=dict(zip(id2w,classes))
    #print(id2w)

    print('Number of Classes {} \n \n  '.format(len(classes)))

    # DATA GENERATORS

    training_set = VideoDataset(train_prefix, train_partition, train_labels, classes ,seq_length,  dim, False, False)
    training_generator = data.DataLoader(training_set, **train_params)

    validation_set = VideoDataset(test_prefix, test_partition, test_labels,classes, seq_length,  dim ,False, False)
    validation_generator = data.DataLoader(validation_set, **test_params)

    # MODEL ILITIALIZE
    model = RNNModel(args.hidden_size,args.n_layers,args.dropout,len(classes)).to(device)
    # model.load_state_dict(
    #     torch.load('/home/hatzis/Desktop/stim_ctc/checkpoints/stim_ctc_firt_try/best_wer46.35898383147232.pth', map_location=device)[
    #         'model_dict'])

    if(args.optim == 1):
        opt_name='ADAMlr '
        print(" use optim ",opt_name,args.lr)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif(args.optim == 2):
        opt_name='SGD lr '
        print(" use optim",opt_name,args.lr)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.8,weight_decay=0.0000001)
    max_epochs = 100


    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # TRAINING LOOP

    for epoch in range(1, max_epochs):

        #scheduler.step()


        train(args, model, device, training_generator, optimizer, epoch, timestamp, ngrams, id2w)

        print("!!!!!!!!   VALIDATION   !!!!!!!!!!!!!!!!!!")
        validate(args, model, device, validation_generator, optimizer, epoch, timestamp, ngrams, id2w)
        #save_checkpoint(for_checkpoint, is_best,'../checkpoints/test_day_'+timestamp )

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
