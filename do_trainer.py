# -*- encoding: utf-8 -*-
import sys
import os
import argparse
from common import *


def parse_args():
    parser = argparse.ArgumentParser('W2VVPP training script.')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('trainCollection', type=str, default='msrvtt10k',
                        help='train collection')
    parser.add_argument('valCollection', type=str, default='tv2016train',
                        help='validation collection')
    parser.add_argument('--trainCollection2', type=str, default='None',
                        help='train collection')
    parser.add_argument('--task2_caption', type=str, default='no_task2_caption',
                        help='the suffix of task2 caption.(It looks like "caption.nouns vocab_nouns") Default is nouns.')
    parser.add_argument('--train_strategy', type=str, default='usual',
                        help='train strategy.("usual, subset") Default is usual.')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--val_set', type=str, default='setA',
                        help='validation collection set (no, setA, setB). (default: setA)')
    parser.add_argument('--metric', type=str, default='mir', choices=['r1', 'r5', 'medr', 'meanr', 'mir'],
                        help='performance metric on validation set')
    parser.add_argument('--num_epochs', default=80, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_prefix', default='runs_0', type=str,
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--config_name', type=str, default='w2vvpp_resnext101-resnet152_subspace',
                        help='model configuration file. (default: w2vvpp_resnext101-resnet152_subspace')
    # parser.add_argument('--model_name', type=str, default='abandoned, Please refer to config.model_name',
    #                     help='The param was abandoned')
    parser.add_argument('--parm_adjust_config', type=str, default='None',
                        help='the config parm you need to set. (default: None')
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument('--random_seed', default=2, type=int,
                        help='random_seed of the trainer')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='distributed rank if use muti-gpu')
    parser.add_argument('--pretrained_file_path', default='None', type=str,
                        help='Whether use previous model to train')
    parser.add_argument('--save_mean_last', default=0, type=int, choices=[0, 1],
                        help='Whether save the average of last 10 epoch model')
    parser.add_argument('--task3_caption', type=str, default='no_task3_caption',
                        help='the suffix of task3 caption.(It looks like "caption.false ") Default is false.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    import torch
    print(torch.cuda.device_count())
    return args


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print()
        # from model.model import get_model, get_we
        # gcc
        # sys.argv = "trainer.py --device 4 gcc11train gcc11val " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 256 " \
        #            "--workers 16 " \
        #            "--train_strategy usual " \
        #            "--model_name w2vpp_mutivis_attention " \
        #            "--config w2vvpp_mutiVisual_subspace_AvsDataset_AdjustCLIP " \
        #            "--parm_adjust_config 0_9_1_9 " \
        #            "--val_set no " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

        # msrvtt with task2
        # sys.argv = "trainer.py --device 3 msrvtt10ktrain msrvtt10kval " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 256 " \
        #            "--workers 10 --task2_caption vg_confidence_thresholdSumRank_freqBt5 " \
        #            "--train_strategy usual " \
        #            "--model_name w2vvpp_wv2object " \
        #            "--config_name w2vvpp_resnext101resnet152_subspace_addPritrainedObject_bow_adjust_alpha " \
        #            "--parm_adjust_config 0 " \
        #            "--val_set no " \
        #            "--pretrained_file_path /data/liupengju/hf/gcc11train/w2vvpp_train/gcc11train/w2vvpp_resnext101resnet152_subspace_addPritrainedObject_bow_adjust_alpha/runs_bow_w2vvpp_wv2object_a_0.5/model_best.pth.tar " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

        # msrvtt w2vvpp_attention
        # sys.argv = "trainer.py --device 4 msrvtt10ktrain msrvtt10kval " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 64 " \
        #            "--train_strategy usual " \
        #            "--config_name experiments.MultiHeadAttention_adjustVisTxt " \
        #            "--parm_adjust_config 0_12_3_12_1_0_1 " \
        #            "--val_set no " \
        #            "--save_mean_last 0 " \
        #            "--pretrained_file_path None " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

        # msrvtt CLIPEnd2End
        # sys.argv = "trainer.py --device 1 msrvtt10ktrain msrvtt10kval_small " \
        #            "--rootpath ~/hf_code/VisualSearch --batch_size 32 " \
        #            "--workers 10 " \
        #            "--config_name CLIP.CLIPEnd2End " \
        #            "--parm_adjust_config 0 " \
        #            "--val_set no " \
        #            "--pretrained_file_path None " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 0".split(' ')

        sys.argv = "trainer.py --device cpu msrvtt10ktrain msrvtt10kval " \
                   "--rootpath ~/hf_code/VisualSearch --batch_size 8 " \
                   "--workers 10 " \
                   "--config_name experiments.MMT " \
                   "--parm_adjust_config 1_2_8_0 " \
                   "--val_set no " \
                   "--pretrained_file_path None " \
                   "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

        # msvd CLIPEnd2End
        # sys.argv = "trainer.py --device 1 msvdtrain msvdval " \
        #            "--rootpath ~/hf_code/VisualSearch --batch_size 32 " \
        #            "--workers 10 " \
        #            "--config_name CLIP.CLIPEnd2End " \
        #            "--parm_adjust_config 0 " \
        #            "--val_set no " \
        #            "--pretrained_file_path None " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')


        # msrvtt w2vpp_MutiVisFrameFeat_attention
        # sys.argv = "trainer.py --device 0 msrvtt10ktrain msrvtt10kval " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 256 " \
        #            "--workers 10 " \
        #            "--train_strategy usual " \
        #            "--model_name w2vpp_MutiVisFrameFeat_attention " \
        #            "--config_name w2vvpp_mutiVisual_subspace_AdjustVisframeEncoder " \
        #            "--parm_adjust_config 0_4_1 " \
        #            "--val_set no " \
        #            "--pretrained_file_path None " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

        # msrvtt sea
        # sys.argv = "trainer.py --device 4 msrvtt10ktrain msrvtt10kval " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 256 " \
        #            "--workers 10 " \
        #            "--train_strategy usual " \
        #            "--config_name AAAI.sea_adjustVisTxt " \
        #            "--parm_adjust_config 1_8_6_8 " \
        #            "--val_set no " \
        #            "--pretrained_file_path None " \
        #            "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

        # tgif w2vvpp_attention
        # sys.argv = "trainer.py --device 1 tgif-msrvtt10k tv2016train " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 128 " \
        #            "--workers 10 --task2_caption nouns " \
        #            "--train_strategy usual " \
        #            "--config_name AAAI.sea_adjustVisTxt " \
        #            "--parm_adjust_config 1_8_0_8 " \
        #            "--val_set setA " \
        #            "--model_prefix test1 --overwrite 1".split(' ')

        # tgif-vatex sea
        # sys.argv = "trainer.py --device 0 tgif-msrvtt10k-vatex tv2016train " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 64 " \
        #            "--workers 10 " \
        #            "--train_strategy usual " \
        #            "--config_name experiments.sea_only_visual_multi_head_avs_adjustVisTxt " \
        #            "--parm_adjust_config 2_11_0_1 " \
        #            "--val_set setA " \
        #            "--model_prefix test1 --overwrite 1".split(' ')

        # tgif-vatex sea_multi_head
        # sys.argv = "trainer.py --device 0 tgif-msrvtt10k-vatex tv2016train " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 64 " \
        #            "--workers 10 " \
        #            "--train_strategy usual " \
        #            "--config_name experiments.sea_only_visual_multi_head_add_concat_avs_adjustVisTxt " \
        #            "--parm_adjust_config 3_11_1_1 " \
        #            "--val_set setA " \
        #            "--model_prefix test1 --overwrite 1".split(' ')

        # coco sea_multi_head
        # sys.argv = "trainer.py --device 0 tgif-msrvtt10k-vatex tv2016train " \
        #            "--rootpath /home/liupengju/hf_code/VisualSearch --batch_size 64 " \
        #            "--workers 10 " \
        #            "--train_strategy usual " \
        #            "--config_name experiments.sea_plus_avs_multi_head_adjustVisTxt " \
        #            "--parm_adjust_config 11_12_1_12 " \
        #            "--val_set setA " \
        #            "--model_prefix test1 --overwrite 1".split(' ')

    opt = parse_args()

    from trainer import main, parse_args, main_subset
    if parse_args().train_strategy == 'subset':
        main_subset(opt)
    elif parse_args().train_strategy == 'usual':
        main(opt)
    else:
        raise Exception("No this train_strategy")