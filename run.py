import argparse
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import sys
import os 
from art import *

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

    parser = argparse.ArgumentParser(description='channelformer')

    # -------------------------------------------- Input and Output --------------------------------------------------------
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of the train-set to be used as validation')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--problem', type=str, required=True, default='test', help='dataset name')
    parser.add_argument('--model', type=str, required=True, default='ChannelFormer', help='model name')
    parser.add_argument('--dataset', type=str, required=True, default='UEA', help='dataset type')
    parser.add_argument('--data_path', type=str, default='./dataset/UWaveGestureLibrary/', help='root path of the data file')

    # -------------------------------------------- Training Parameters/ Hyper-Parameters ------------------------------------
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads for encoder')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--data_use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    parser.add_argument('--c_n_heads', type=int, default=8, help='num of heads for channel attention')
    parser.add_argument('--c_d_model', type=int, default=2048, help='dimension of channel attention')
    parser.add_argument('--train_id', type=str, default='2024', help='train id')

    # ----------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ System --------------------------------------------------------
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=1234, help="Randomization seed")

    args = parser.parse_args()

    # -------------------------------------------------- GPU and Ouput SETTING ----------------------------------------------------
    args.use_gpu = True if torch.cuda.is_available() else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.order_input   = ' '.join(sys.argv)
    args.results_path  = os.path.join('./results', args.train_id + '_' + args.model, args.problem)
    args.checkpoints   = os.path.join(args.results_path, 'checkpoints')
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        os.makedirs(args.checkpoints)

    # ---------------------------------------------- Training and Testing Module ------------------------------------
    print(text2art(args.problem, font='small'))
    print('results_path: ', args.results_path)
    Exp = Exp_Classification
    setting = '{}_{}_bs{}_dm{}_dff{}_cdm{}_nh{}_cnh{}_el{}_datanorm{}'.format(
            args.problem,
            args.model,
            args.batch_size,
            args.d_model,
            args.d_ff, 
            args.c_d_model,
            args.n_heads,
            args.c_n_heads,
            args.e_layers,
            args.data_use_norm,
        )

    if args.is_training:
        exp = Exp(args)  # set experiments

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    