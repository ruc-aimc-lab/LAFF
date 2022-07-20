# -*-coding:utf-8 -*-
import sys
import time
import json
import shutil
import importlib

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing

import util
import evaluation
import data_provider as data
from common import *
from bigfile import BigFile
from txt2vec import get_txt2vec
from generic_utils import Progbar
from model.model import get_model, get_we
from do_trainer import parse_args
from collections import OrderedDict


def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.config()


def load_pretrained_model(pretrained_file_path, rootpath, device):
    checkpoint = torch.load(pretrained_file_path, map_location='cpu')
    epoch = checkpoint['epoch']
    best_perf = checkpoint['best_perf']
    config = checkpoint['config']
    # config.lr = config.lr / 10  # 进行学习率调整

    model_name = config.model_name
    if hasattr(config, 't2v_w2v'):
        # w2v
        w2v_data_path = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m')
        config.t2v_w2v = get_txt2vec("w2v_nsw")(w2v_data_path)

        config.we = get_we(config.t2v_idx.vocab, config.t2v_w2v.data_path)

    # Construct the model
    model = get_model(model_name, device, config)
    model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(("=> loaded checkpoint '{}' (epoch {}, best_perf {})"
          .format(pretrained_file_path, epoch, best_perf)))

    return {'model': model, 'config': config}


def prepare_config(opt, checkToSkip=True):
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if '~' in opt.rootpath:
        opt.rootpath = opt.rootpath.replace('~', os.path.expanduser('~'))
    rootpath = opt.rootpath

    trainCollection = opt.trainCollection
    if "trainCollection2" in opt:
        trainCollection2 = opt.trainCollection2
    else:
        trainCollection2 ="None"
    valCollection = opt.valCollection
    task2_caption_suffix = opt.task2_caption  # 提取的标签的文件后缀
    if "task3_caption" in opt:
        task3_caption_suffix = opt.task3_caption
    else:
        task3_caption_suffix='no_task3_caption'
    if opt.val_set == 'no':
        val_set = ''
    else:
        val_set = opt.val_set

    # cuda number
    global device
    if torch.cuda.is_available() and opt.device != "cpu":
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # set the config parm you adjust
    config = load_config('configs.%s' % opt.config_name)  # 模型参数文件
    if opt.parm_adjust_config != 'None':
        config.adjust_parm(opt.parm_adjust_config)
    if trainCollection2 == "None":
        model_path = os.path.join(rootpath, trainCollection, 'w2vvpp_train', valCollection, val_set, opt.config_name,
                              opt.model_prefix)
    else:
        model_path = os.path.join(rootpath, trainCollection+'_'+trainCollection2, 'w2vvpp_train', valCollection, val_set, opt.config_name,
                              opt.model_prefix)
    if checkToSkip:
        if util.checkToSkip(os.path.join(model_path, 'model_best.pth.tar'), opt.overwrite):
            sys.exit(0)
        util.makedirs(model_path)

        print(json.dumps(vars(opt), indent=2))

    model_name = config.model_name

    global writer
    writer = SummaryWriter(log_dir=model_path, flush_secs=5)

    collections = {'train': trainCollection, 'val': valCollection}  # 数据集

    # 标注文件
    capfiles = {'train': '%s.caption.txt', 'val': os.path.join(val_set, '%s.caption.txt')}

    if trainCollection2 != 'None':
        collections['train2']=trainCollection2
        capfiles['train2']='%s.caption.txt'
        vocabsuffix=trainCollection+"_"+trainCollection2
    else:
        vocabsuffix = trainCollection

    # 标注文件
    cap_file_paths = {x: os.path.join(rootpath, collections[x], 'TextData', capfiles[x] % collections[x]) for x in
                      collections}
    capfiles_negationset = os.path.join(val_set, '%s.caption.negationset.txt'% collections["val"])
    capfiles_negationset = os.path.join(rootpath,valCollection, 'TextData', capfiles_negationset)
    config.capfiles_negationset = capfiles_negationset
    # ***************************萌萌哒*****************************
    # 视频 Feature 文件
    vis_feat_files = {x: None for x in collections}
    if len(config.vid_feats) > 0:
        vis_feat_files = {collection: {y: BigFile(os.path.join(rootpath, collections[collection], 'FeatureData', y))
                                       for y in config.vid_feats} for collection in collections}
        # config.vis_fc_layers = list(map(int, config.vis_fc_layers.split('-')))
        config.vis_fc_layers[0] = {}
        for each in vis_feat_files['train'].keys():
            config.vis_fc_layers[0][each] = vis_feat_files['train'][each].ndims
        if config.vis_feat_add_concat:
            feat_dim_sum = np.sum(list(config.vis_fc_layers[0].values()))
            config.vis_fc_layers[0]['vis_feat_add_concat'] = feat_dim_sum

    # 视频 muti_feat 文件 （Faster-rnn 特征）
    vis_muti_feat_dicts = {x: None for x in collections}
    if config.SGRAF:
        vis_muti_feat_paths = {x: os.path.join(rootpath, collections[x], 'VideoMultiLabelFeat', config.muti_feat) for x
                               in
                               collections}
        if os.path.realpath(vis_muti_feat_paths['train']) == os.path.realpath(vis_muti_feat_paths['val']):
            vis_muti_feat_dicts['train'] = vis_muti_feat_dicts['val'] = np.load(vis_muti_feat_paths['train'],
                                                                                allow_pickle=True).item()
        else:
            vis_muti_feat_dicts['train'] = np.load(vis_muti_feat_paths['train'], allow_pickle=True).item()
            vis_muti_feat_dicts['val'] = np.load(vis_muti_feat_paths['val'], allow_pickle=True).item()

    # 视频帧特征文件
    vis_frame_feat_dicts = {x: None for x in collections}
    if config.frame_feat_input:
        vis_frame_feat_dicts = {
            collection: {y: BigFile(os.path.join(rootpath, collections[collection], 'FeatureData/frame', y))
                         for y in config.vid_frame_feats} for collection in collections}
        for each in vis_frame_feat_dicts['train'].keys():  # 增加相关维度信息
            config.vis_fc_layers[0][each] = vis_frame_feat_dicts['train'][each].ndims
    # 视频帧文件
    if config.frame_loader:
        frame_id_path_file = {'train': os.path.join(rootpath, trainCollection, 'id.imagepath.txt'),
                              'val': os.path.join(rootpath, valCollection, 'id.imagepath.txt')
                              }
    else:
        frame_id_path_file = {'train': None,
                              'val': None
                              }

    # ***************************萌萌哒*****************************
    # 各种 text encode
    if type(config.text_encoding['bow_encoding']) is str:
        for name in config.text_encoding:
            encoding = config.text_encoding[name]
            config.text_encoding[name] = {}
            config.text_encoding[name]["name"] = encoding
    bow_encoding, w2v_encoding, rnn_encoding = (
        config.text_encoding['bow_encoding']['name'],
        config.text_encoding['w2v_encoding']['name'],
        config.text_encoding['rnn_encoding']['name'],
    )
    rnn_encoding, config.pooling = rnn_encoding.split('_', 1)
    # bow
    if 'no' in bow_encoding:
        config.t2v_bow = No()
        config.t2v_bow.ndims = 0
    else:
        bow_vocab_file = os.path.join(rootpath, vocabsuffix, 'TextData', 'vocab',
                                      '%s_%d.pkl' % (bow_encoding, config.threshold))
        config.t2v_bow = get_txt2vec(bow_encoding)(bow_vocab_file, norm=config.bow_norm)
    # w2v
    w2v_data_path = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m')
    config.t2v_w2v = get_txt2vec("w2v_nsw")(w2v_data_path)
    # gru
    rnn_vocab_file = os.path.join(rootpath, vocabsuffix, 'TextData', 'vocab',
                                  '%s_%d.pkl' % ('gru', config.threshold))
    if 'bigru' == rnn_encoding:
        # 如果是 bidirectional 那么就把 rnn_size × 2，让全连接层接受到，之后再把 rnn_size / 2，因为之后建立模型时还要再乘2
        config.rnn_size *= 2
    elif rnn_encoding == 'nogru':
        config.rnn_size = 0
    elif rnn_encoding == 'gru':

        rnn_vocab_file = os.path.join(rootpath, vocabsuffix, 'TextData', 'vocab',
                                      '%s_%d.pkl' % (rnn_encoding, config.threshold))

    else:
        raise Exception('No this gru type!')

    config.t2v_idx = get_txt2vec('idxvec')(rnn_vocab_file)
    if config.we_dim == 500:
        config.we = get_we(config.t2v_idx.vocab, w2v_data_path)
    # txt 全连接层
    config.txt_fc_layers = list(map(int, config.txt_fc_layers.split('-')))
    if 'bigru' in rnn_encoding:
        config.rnn_size = config.rnn_size // 2

    # **********************************萌萌哒分界线********************************************
    # 任务2：对 object_caption 编码
    if task2_caption_suffix == 'no_task2_caption':
        cap_file_paths_task2 = {x: None for x in collections}
    else:
        capfiles_task2 = {'train': '%s.caption.%s.txt' % ('%s', task2_caption_suffix),
                          'val': os.path.join(val_set, '%s.caption.%s.txt' % ('%s', task2_caption_suffix))}
        cap_file_paths_task2 = {
            x: os.path.join(rootpath, collections[x], 'TextData', capfiles_task2[x] % collections[x])
            for x in
            collections}

        # 各种 text encode
        bow_encoding_task2 = config.text_encoding_task2

        bow_vocab_file_task2 = os.path.join(rootpath, trainCollection, 'TextData', 'vocab_%s' % task2_caption_suffix,
                                            '%s_%d.pkl' % (bow_encoding_task2, config.threshold_task2))
        config.t2v_bow_task2 = get_txt2vec(bow_encoding_task2)(bow_vocab_file_task2, norm=config.bow_norm_task2)
        print(config.t2v_bow_task2)

        # 视频转换参数
        config.vis_fc_layers_task2 = list(map(int, config.vis_fc_layers_task2.split('-')))
        config.vis_fc_layers_task2[0] = config.vis_fc_layers[0]
        config.vis_fc_layers_task2[1] = config.t2v_bow_task2.ndims

        # 文本转换参数
        config.txt_fc_layers_task2 = list(map(int, config.txt_fc_layers_task2.split('-')))
        if config.txt_feature_task2 == 'bow':
            config.txt_fc_layers_task2[0] = config.t2v_bow.ndims
        elif config.txt_feature_task2 == 'w2v':
            config.txt_fc_layers_task2[0] = config.t2v_w2v.ndims
        elif config.txt_feature_task2 == 'gru':
            config.txt_fc_layers_task2[0] = 2 * config.rnn_size if rnn_encoding == 'bigru' else config.rnn_size
        elif config.txt_feature_task2 == 'no':
            pass
        else:
            raise Exception('No this txt_feature_task2 implement!')
        config.txt_fc_layers_task2[1] = config.t2v_bow_task2.ndims
    if task3_caption_suffix == 'no_task3_caption':
        config.task3 = False
        cap_file_paths_task3 = {x: None for x in collections}
    else:
        config.task3=True
        capfiles_task3 = {'train': '%s.caption.%s.txt' % ('%s', task3_caption_suffix),
                          'val': os.path.join(val_set, '%s.caption.%s.txt' % ('%s', task3_caption_suffix))}
        cap_file_paths_task3 = {
        x: os.path.join(rootpath, collections[x], 'TextData', capfiles_task3[x] % collections[x])
        for x in collections}


    if opt.pretrained_file_path != 'None':
        pretrained_model = load_pretrained_model(opt.pretrained_file_path, opt.rootpath, device)
        config.t2v_bow = pretrained_model['config'].t2v_bow
        config.t2v_idx = pretrained_model['config'].t2v_idx
        config.we = pretrained_model['config'].we
        model = pretrained_model['model']
    else:
        model = get_model(model_name, device, config)


    prepared_configs = {'vis_feat_files': vis_feat_files,
                        'vis_muti_feat_dicts': vis_muti_feat_dicts,
                        'vis_frame_feat_dicts': vis_frame_feat_dicts,
                        'frame_id_path_file': frame_id_path_file,
                        'cap_file_paths': cap_file_paths,
                        'cap_file_paths_task2': cap_file_paths_task2,
                        'cap_file_paths_task3': cap_file_paths_task3,
                        'opt': opt,
                        'val_set': val_set,
                        'config': config,
                        'collections': collections,
                        'model_path': model_path,
                        'device': device,
                        'task2_caption_suffix': task2_caption_suffix,
                        'task3_caption_suffix': task3_caption_suffix,
                        'capfiles_negationset': capfiles_negationset,
                        'model': model,
                        }
    return prepared_configs

def prepare_model1(opt):
    prepared_configs = prepare_config(opt)
    config = prepared_configs['config']
    model_name = config.model_name

    if opt.pretrained_file_path != 'None':
        pretrained_model = load_pretrained_model(opt.pretrained_file_path, opt.rootpath, device)
        config = pretrained_model['config']
        model = pretrained_model['model']
    else:
        model = get_model(model_name, device, config)

    model = model.to(device)


    prepared_configs['model'] = model
    return prepared_configs


def main(opt):
    prepared_configs = prepare_config(opt)
    vis_feat_files = prepared_configs['vis_feat_files']
    vis_frame_feat_dicts = prepared_configs['vis_frame_feat_dicts']
    frame_id_path_file = prepared_configs['frame_id_path_file']
    vis_muti_feat_dicts = prepared_configs['vis_muti_feat_dicts']
    cap_file_paths = prepared_configs['cap_file_paths']
    cap_file_paths_task2 = prepared_configs['cap_file_paths_task2']
    cap_file_paths_task3= prepared_configs['cap_file_paths_task3']
    opt = prepared_configs['opt']
    config = prepared_configs['config']
    collections = prepared_configs['collections']
    model_path = prepared_configs['model_path']
    model = prepared_configs['model']
    device = prepared_configs['device']
    val_set = prepared_configs['val_set']

    vis_ids = list(
        map(str.strip, open(os.path.join(opt.rootpath, opt.trainCollection, 'VideoSets', opt.trainCollection + '.txt'))))
    data_loaders = {x: data.pair_provider({'vis_feat_files': vis_feat_files[x], 'capfile': cap_file_paths[x],
                                           'vis_frame_feat_dicts': vis_frame_feat_dicts[x],
                                           'vis_ids': vis_ids,
                                           'max_frame': config.max_frame,
                                           'sample_type': config.frame_sample_type_train,
                                           'vis_muti_feat_dicts': vis_muti_feat_dicts[x],
                                           'frame_id_path_file': frame_id_path_file['train'],
                                           'capfile_task2': cap_file_paths_task2[x],
                                           'capfile_task3': cap_file_paths_task3[x], 'pin_memory': False,
                                           'batch_size': opt.batch_size, 'num_workers': opt.workers,
                                           'config': config,
                                           'collection': x,
                                           'shuffle': (x == 'train'), 'task3': config.task3
                                           })
                    for x in collections}

    vis_ids = list(map(str.strip, open(os.path.join(opt.rootpath, opt.valCollection, 'VideoSets', opt.valCollection + '.txt'))))
    vis_loader_val = data.vis_provider({'vis_feat_files': vis_feat_files['val'], 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts['val'],
                                    'max_frame': config.max_frame,
                                        'sample_type': config.frame_sample_type_test,
                                        'frame_id_path_file': frame_id_path_file['val'],
                                    'batch_size': int(opt.batch_size * 2),
                                        'config': config,
                                        'num_workers': opt.workers})
    capfile = os.path.join(opt.rootpath, opt.valCollection, 'TextData', val_set, opt.valCollection+'.caption.txt')
    txt_loader_val = data.txt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                    'batch_size': int(opt.batch_size * 2),
                                        'num_workers': opt.workers,
                                        'task3':config.task3})

    # Train the Model
    best_perf = 0
    no_impr_counter = 0
    val_perf_hist_fout = open(os.path.join(model_path, 'val_perf_hist.txt'), 'w')

    save_checkpoint({'epoch': 0 + 1, 'model': model.state_dict(), 'best_perf': best_perf,
                     'config': config, 'opt': opt}, True, logdir=model_path, only_best=False,
                    filename='checkpoint_epoch_%s.pth.tar' % 0)
    for epoch in range(opt.num_epochs):
        logger.info(json.dumps(vars(opt), indent=2))
        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, model.learning_rate))
        print('-' * 10)

        writer.add_scalar('train/learning_rate', model.learning_rate[0], epoch)

        # if epoch > 0 and hasattr(model, 'change_raw_global_emb_weight'):
        if hasattr(model, 'change_raw_global_emb_weight'):
            model.change_raw_global_emb_weight()
        # train for one epoch
        train(model, data_loaders['train'], epoch)
        # additional training data
        if 'train2' in data_loaders:
            train(model, data_loaders['train2'], epoch)
        # evaluate on validation set

        cur_perf2 = 0
        cur_perf, cur_perf2 = validate(model, txt_loader_val, vis_loader_val, epoch, measure=config.measure, metric=opt.metric,
                            config=config, negative_val=config.task3)

        model.lr_step(val_value=cur_perf)

        print(' * Current perf: {}\n * Best perf: {}\n'.format(cur_perf, best_perf))
        val_perf_hist_fout.write('epoch_%d:\nText2Video(%s): %f\n' % (epoch, opt.metric, cur_perf))
        val_perf_hist_fout.flush()

        # remember best performance and save checkpoint
        is_best = cur_perf > best_perf
        best_perf = max(cur_perf, best_perf)
        config.t2v_w2v = None
        save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(), 'best_perf': best_perf,
                         'config': config, 'opt': opt}, is_best, logdir=model_path, only_best=False,
                        filename='checkpoint_epoch_%s.pth.tar' % epoch)
        if is_best:
            no_impr_counter = 0
            model_dict = None
        elif opt.save_mean_last == 1:
            if model_dict is None:
                model_dict = model.state_dict()
                worker_state_dict = [model_dict]
            else:
                worker_state_dict.append(model.state_dict())
                weight_keys = list(worker_state_dict[0].keys())
                fed_state_dict = OrderedDict()
                for key in weight_keys:
                    key_sum = 0
                    for i in range(len(worker_state_dict)):
                        key_sum = key_sum + worker_state_dict[i][key]
                    fed_state_dict[key] = key_sum / len(worker_state_dict)
                torch.save({'epoch': epoch + 1, 'model': fed_state_dict, 'best_perf': best_perf,
                            'config': config, 'opt': opt}, os.path.join(model_path, 'mean_last10.pth.tar'))

        no_impr_counter += 1
        if no_impr_counter > 10 or epoch == opt.num_epochs-1:
            save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(), 'best_perf': best_perf,
                             'config': config, 'opt': opt}, is_best=False, logdir=model_path, only_best=True,
                            filename='checkpoint_epoch_%s.pth.tar' % epoch)
            print('Early stopping happended or stopped.\n')
            print(json.dumps(vars(opt), indent=2))
            break

        # 测试状态下早停
        if __name__ == '__main__' and epoch > 1:
            break

    val_perf_hist_fout.close()
    message = 'best performance on validation:\n Text to video({}): {}'.format(opt.metric, best_perf)
    print(message)
    with open(os.path.join(model_path, 'val_perf.txt'), 'w') as fout:
        fout.write(message)


def get_negationset(capfile):
    negationset=set()
    with open(capfile, 'r') as reader:
        lines = reader.readlines()

        for line in lines:
            cap_id, caption = line.strip().split(' ', 1)
            negationset.add(cap_id)
    return negationset

def main_subset(opt):
    prepared_configs = prepare_config(opt)
    vis_feat_files = prepared_configs['vis_feat_files']
    cap_file_paths = prepared_configs['cap_file_paths']
    cap_file_paths_task2 = prepared_configs['cap_file_paths_task2']
    opt = prepared_configs['opt']
    config = prepared_configs['config']
    collections = prepared_configs['collections']
    model_path = prepared_configs['model_path']
    model = prepared_configs['model']
    device = prepared_configs['device']
    task2_caption_suffix = prepared_configs['task2_caption_suffix']

    # 划分 data loader，shuffle 一定为 True
    params = {'vis_feat': vis_feat_files['train'], 'capfile': cap_file_paths['train'],
              'capfile_task2': cap_file_paths_task2['train'], 'pin_memory': False,
              'batch_size': opt.batch_size, 'sampler': None, 'num_workers': opt.workers,
              'shuffle': True, 'task2_caption_suffix': task2_caption_suffix}
    data_loader_old = data.pair_provider(params)
    data_induce = np.arange(0, data_loader_old.dataset.length)
    data_loaders = {}
    train_val_split = int(0.985 * data_loader_old.dataset.length)
    if __name__ != '__main__' and torch.cuda.device_count() > 1:
        params['sampler'] = 'NotNone'
        params['shuffle'] = False
    data_loaders['train'] = data.pair_provider_subset(params, data_induce[0:train_val_split])
    data_loaders['val'] = data.pair_provider_subset(params, data_induce[train_val_split:])  # 不能太大，否则爆显存

    # Train the Model
    best_perf = 0
    no_impr_counter = 0
    val_perf_hist_fout = open(os.path.join(model_path, 'val_perf_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):
        logger.info(json.dumps(vars(opt), indent=2))
        # if "," in opt.device:
        #     model1 = model
        #     model = model.module
        # else:
        #     model1 = model
        print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, model.learning_rate))
        print('-' * 10)

        writer.add_scalar('train/learning_rate', model.learning_rate[0], epoch)
        # train for one epoch
        train(model, data_loaders['train'], epoch)

        # evaluate on validation set
        cur_perf = validate(model, data_loaders['val'], epoch, measure=config.measure, metric=opt.metric,
                            config=config)
        model.lr_step(val_value=cur_perf)

        print(' * Current perf: {}\n * Best perf: {}\n'.format(cur_perf, best_perf))
        val_perf_hist_fout.write('epoch_%d:\nText2Video(%s): %f\n' % (epoch, opt.metric, cur_perf))
        val_perf_hist_fout.flush()

        # remember best performance and save checkpoint
        is_best = cur_perf > best_perf
        best_perf = max(cur_perf, best_perf)
        save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(), 'best_perf': best_perf,
                         'config': config, 'opt': opt}, is_best, logdir=model_path, only_best=True,
                        filename='checkpoint_epoch_%s.pth.tar' % epoch)
        if is_best:
            no_impr_counter = 0
        else:
            no_impr_counter += 1
            if no_impr_counter > 9:
                print('Early stopping happended.\n')
                print(json.dumps(vars(opt), indent=2))
                break

    val_perf_hist_fout.close()
    message = 'best performance on validation:\n Text to video({}): {}'.format(opt.metric, best_perf)
    print(message)
    with open(os.path.join(model_path, 'val_perf.txt'), 'w') as fout:
        fout.write(message)


def train(model, train_loader, epoch):
    # average meters to record the training statistics
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()

    # switch to train mode
    model.train()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()

    for i, train_data in enumerate(train_loader):
        if __name__ == '__main__':
            pass
            if i > 5:
                break
                # sys.exit(0)

            # progbar.add(len(train_data['idxs']))
            # continue

        data_time.update(time.time() - end)

        input_idxs = train_data['idxs']
        loss_items = model(train_data, epoch)

        values = [('batch_time', batch_time.val)]
        # print(loss_items)
        # print(torch.cuda.is_available())
        for key in loss_items.keys():
            if isinstance(loss_items[key], torch.Tensor):
                loss_items[key] = round(loss_items[key].item(), 4)
            values.append((key, loss_items[key]))
        progbar.add(len(input_idxs), values=values)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        writer.add_scalar('train/Loss', sum(list(loss_items.values())), model.iters)
        for key in loss_items.keys():
            writer.add_scalar('train/'+key, loss_items[key], model.iters)
    print()


def validate(model, txt_loader, vis_loader, epoch, measure='cosine', metric='mir',negative_val=False, config=None):
    # compute the encoding for all the validation videos and captions
    # vis_embs: 200*2048,  txt_embs: 200*2048, vis_ids: 200, txt_ids: 200
    txt2vis_sim, txt_ids, vis_ids = model.predict(txt_loader, vis_loader, measure=config.measure)

    inds = np.argsort(txt2vis_sim, axis=1)
    label_matrix = np.zeros(inds.shape)  #
    if negative_val:
        negative_index=[]
        capfiles_negationset = get_negationset(config.capfiles_negationset)

    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        # print(txt_ids[index])
        gt_index = np.where(np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]
        label_matrix[index][gt_index] = 1
        if negative_val:
            if txt_ids[index] in capfiles_negationset:
                negative_index.append(index)

    (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
    write_metric(r1, r5, r10, medr, meanr, mir, mAP, epoch)
    mir2 = None
    if negative_val:
        (r1, r5, r10, medr, meanr, mir2, mAP) = evaluation.eval(label_matrix[negative_index])
        print("negtive_set")
        write_metric(r1, r5, r10, medr, meanr, mir2, mAP, epoch, mode="task3")

    return locals().get(metric, mir), locals().get("mir2", mir2)


def write_metric(r1, r5, r10, medr, meanr, mir, mAP, epoch, mode="task1"):
    sum_recall = r1 + r5 + r10
    print(" * Text to video:")
    print(" * r_1_5_10: {}".format([round(r1, 3), round(r5, 3), round(r10, 3)]))
    print(" * medr, meanr, mir: {}".format([round(medr, 3), round(meanr, 3), round(mir, 3)]))
    print(" * mAP: {}".format(round(mAP, 3)))
    print(" * " + '-' * 10)
    writer.add_scalar(mode + 'val/r1', r1, epoch)
    writer.add_scalar(mode + 'val/r5', r5, epoch)
    writer.add_scalar(mode + 'val/r10', r10, epoch)
    writer.add_scalar(mode + 'val/medr', medr, epoch)
    writer.add_scalar(mode + 'val/meanr', meanr, epoch)
    writer.add_scalar(mode + 'val/mir', mir, epoch)
    writer.add_scalar(mode + 'val/mAP', mAP, epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', only_best=False, logdir=''):
    """

    :param state:
    :param is_best: 比以前的好，就保存下来
    :param filename:
    :param only_best: 当结束训练时，only_best=True, 删除 checkpoint.pth.tar 文件，把 model_temp_best.pth.tar 文件 复制成 model_best.pth.tar
    :param logdir:
    :return:
    """
    resfile = os.path.join(logdir, filename)

    if is_best:
        torch.save(state, resfile)
        shutil.copyfile(resfile, os.path.join(logdir, 'model_temp_best.pth.tar'))
        os.remove(resfile)

    if only_best:
        shutil.copyfile(os.path.join(logdir, 'model_temp_best.pth.tar'), os.path.join(logdir, 'model_best.pth.tar'))
        os.remove(os.path.join(logdir, 'model_temp_best.pth.tar'))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print()
        # msrvtt clip
        sys.argv = "trainer.py --device 1 msrvtt10ktrain msrvtt10kval " \
                   "--rootpath ~/VisualSearch --batch_size 128 " \
                   "--workers 2 " \
                   "--train_strategy usual " \
                   "--config_name CVPR.FrameLaff_NoFrameFc_StrongCLIP_adjust " \
                   "--parm_adjust_config 0_7_1_12_0_12_0 " \
                   "--val_set no " \
                   "--pretrained_file_path None " \
                   "--model_prefix bow_w2v_runs_test1 --overwrite 1".split(' ')

    opt = parse_args()  # Here opt is the input value, config is read in the parameter file
    if parse_args().train_strategy == 'subset':
        main_subset(opt)
    elif parse_args().train_strategy == 'usual':
        main(opt)
    else:
        raise Exception("No this train_strategy")
