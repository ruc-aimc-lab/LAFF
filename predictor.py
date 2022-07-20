# coding=utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import sys
import time
import json
import argparse
import pickle

import numpy as np
import torch

import util
import evaluation
import data_provider as data
import trainer
from common import *
from model.model import get_model
from bigfile import BigFile
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('W2VVPP predictor')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('testCollection', type=str,
                        help='test collection')
    parser.add_argument('model_path', type=str,
                        help='Path to load the model.')
    parser.add_argument('sim_name', type=str,
                        help='sub-folder where computed similarities are saved')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',
                        help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18 and tv19.avs.txt for TRECVID19.')
    parser.add_argument('--predict_result_file', type=str, default='result_log/result_test.txt',
                        help='if dataset=msrvtt10k, print the result to txt_file')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of a predicting mini-batch')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of data loader workers.')
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument('--adjust_weight_predict', type=bool, default=False,
                        help='whether adjust the weight')
    parser.add_argument('--task3_caption', type=str, default='no_task3_caption',
                        help='the suffix of task3 caption.(It looks like "caption.false ") Default is false.')

    args = parser.parse_args()
    return args


def txt2video_write_to_file(pred_result_file, inds, vis_ids, txt_ids, t2i_matrix, pkl_saved_file=None, txt_loader=None, Threshold=2000):

    if len(vis_ids) >= Threshold:  # 只保存前 1e4 的检索结果
        TopK = Threshold
    else:
        TopK = -1
    start = time.time()

    shot_dict = {}  # 写到字典，方便做 demo
    if pred_result_file is not None:
        with open(pred_result_file, 'w') as fout:
            for index in tqdm(range(inds.shape[0])):
                ind = inds[index][::-1][0:TopK]
                fout.write(txt_ids[index] + ' ' + ' '.join([vis_ids[i] + ' %s' % t2i_matrix[index][i]
                                                            for i in ind]) + '\n')
                if pkl_saved_file is not None:
                    shot_dict[txt_ids[index]] = {}
                    shot_dict[txt_ids[index]]['query'] = \
                        txt_loader.dataset.get_caption_dict_by_id(txt_ids[index])["caption"]
                    shot_dict[txt_ids[index]]['rank_list'] = [vis_ids[i] for i in ind]
                    shot_dict[txt_ids[index]]['sim_value'] = [t2i_matrix[index][i] for i in ind]

    if pkl_saved_file is not None:
        if len(shot_dict) == 0:
            for index in tqdm(range(inds.shape[0])):
                ind = inds[index][::-1][0:TopK]
                if pkl_saved_file is not None:
                    shot_dict[txt_ids[index]] = {}
                    shot_dict[txt_ids[index]]['query'] = \
                    txt_loader.dataset.get_caption_dict_by_id(txt_ids[index])["caption"]
                    shot_dict[txt_ids[index]]['rank_list'] = [vis_ids[i] for i in ind]
                    shot_dict[txt_ids[index]]['sim_value'] = [t2i_matrix[index][i] for i in ind]
        with open(pkl_saved_file, 'wb') as f_shot_dict:
            pickle.dump(shot_dict, f_shot_dict)
    print('writing result into file time: %.3f seconds\n' % (time.time() - start))
    print("Save to ", pkl_saved_file)


def write_to_predict_result_file(
        predict_result_file, model_path, checkpoint,
        result_tuple, name_str="Text to video"
                                 ):
    """

    :param predict_result_file:
    :param model_path:
    :param checkpoint:
    :param result_tuple: [(r1, r5, r10, medr, meanr, mir, mAP), ...]
    :return:
    """
    result_file_dir = os.path.dirname(predict_result_file)

    print("pkl result_file_dir: ", predict_result_file)
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir)

    with open(predict_result_file, 'a') as f:

        (r1, r5, r10, medr, meanr, mir, mAP) = result_tuple
        tempStr = " * %s:\n" % name_str
        tempStr += " * r_1_5_10: {}\n".format([round(r1, 3), round(r5, 3), round(r10, 3)])
        tempStr += " * medr, meanr, mir: {}\n".format([round(medr, 3), round(meanr, 3), round(mir, 3)])
        tempStr += " * mAP: {}\n".format(round(mAP, 3))
        tempStr += " * " + '-' * 10
        print(tempStr)

        f.write(str(time.asctime(time.localtime(time.time()))) + '\t')
        for each in [model_path, round(r1, 3), round(r5, 3), round(r10, 3),
                     round(medr, 3), round(meanr, 3), round(mir, 3), round(mAP, 3)]:
            f.write(str(each))
            f.write('\t')
        f.write(checkpoint['opt'].parm_adjust_config.replace('_', '\t'))
        f.write('\n')
    pass


def get_predict_file(opt, checkpoint):
    rootpath = opt.rootpath
    testCollection = opt.testCollection
    # cuda number
    device = torch.device("cuda:{}".format(opt.device)
                          if (torch.cuda.is_available() and opt.device != "cpu") else "cpu")

    resume_file = os.path.join(opt.model_path)

    # Load checkpoint
    epoch = checkpoint['epoch']
    best_perf = checkpoint['best_perf']
    config = checkpoint['config']
    model_name = checkpoint['config'].model_name
    if opt.task3_caption == "no_task3_caption":
        task3 = False
    else:
        task3 = True
    if hasattr(config, 't2v_w2v') and hasattr(config.t2v_w2v, 'w2v'):
        w2v_feature_file = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m', 'feature.bin')
        config.t2v_w2v.w2v.binary_file = w2v_feature_file

    # Construct the model
    model = get_model(model_name, device, config)
    model = model.to(device)
    # print(model)
    # calculate the number of parameters
    try:
        vis_net_params = sum(p.numel() for p in model.vis_net.parameters())
        txt_net_params = sum(p.numel() for p in model.txt_net.parameters())
        print('    VisNet params: %.2fM' % (vis_net_params / 1000000.0))
        print('    TxtNet params: %.2fM' % (txt_net_params / 1000000.0))
        print('    Total params: %.2fM' %
              ((vis_net_params + txt_net_params) / 1000000.0))
    except:
        pass

    # load the checkpoint
    model.load_state_dict(checkpoint['model'], strict=False)
    print("=> loaded checkpoint '{}' (epoch {}, best_perf {})"
          .format(resume_file, epoch, best_perf))
    if "StrongCLIP" in str(checkpoint['config']):
        try:
            # if 'clip_finetune_8frame_uniform_1103' == checkpoint['config'].text_encoding['CLIP_encoding']['dir_name']:
            if "StrongCLIP" in str(checkpoint['config']):
                print("load CLIP-FT model")
                checkpoint1 = torch.load(
                    os.path.join(rootpath, testCollection, 'TextData/clip_finetune_8frame_uniform_1103/model_best.pth.tar'),
                    map_location='cpu')
                import collections
                checkpoint1['model'] = collections.OrderedDict([(k[11:], v) for k, v in checkpoint1['model'].items()])
                model.txt_net.encoder.CLIP_encoder.load_state_dict(checkpoint1['model'], strict=True)

                checkpoint['config'].text_encoding['CLIP_encoding']['dir_name'] = ''
        except Exception as e:
            print("load CLIP-FT model failed!!!")
            print(e)


    vis_feat_files = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData', y))
                                   for y in config.vid_feats}
    # 视频帧特征文件
    vis_frame_feat_dicts = None
    if config.frame_feat_input:
        vis_frame_feat_dicts = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData/frame', y))
                                             for y in config.vid_frame_feats}
    vis_ids = list(map(str.strip, open(os.path.join(rootpath, testCollection, 'VideoSets', testCollection + '.txt'))))
    # 视频帧文件
    if config.frame_loader:
        frame_id_path_file = os.path.join(rootpath, testCollection, 'id.imagepath.txt')
    else:
        frame_id_path_file = None

    vis_loader = data.vis_provider({'vis_feat_files': vis_feat_files, 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts,
                                    'max_frame': config.max_frame,
                                    'sample_type': config.frame_sample_type_test,
                                    'config': config,
                                    'frame_id_path_file': frame_id_path_file,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

    for query_set in opt.query_sets.split(','):
        output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.sim_name)
        pred_result_file = os.path.join(output_dir, 'id.sent.score.txt')

        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        util.makedirs(output_dir)


        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        # load text data
        txt_loader = data.txt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                        'batch_size': opt.batch_size, 'num_workers': opt.num_workers, 'task3': task3})

        t2i_matrix, txt_ids, vis_ids = model.predict(txt_loader, vis_loader, measure=config.measure, record_emb=True)
        # if __name__ == '__main__':
        #     import torch.nn.functional as F
        #     t2i_matrix = torch.Tensor(t2i_matrix)
        #     t2i_matrix = F.softmax(t2i_matrix, dim=1) * F.softmax(t2i_matrix, dim=0)
        #     t2i_matrix = t2i_matrix.numpy()
        #     print("dual softmax inference")

        inds = np.argsort(t2i_matrix, axis=1)

        if testCollection not in ['iacc.3', 'v3c1'] and query_set != "simple_query.txt":

            # caption2index 里面是 ('video001#1', caption, 1, [video001, ...])，这样的 caption 到 gt 检索结果的形式，最后是前10个结果。
            caption2index = []
            label_matrix = np.zeros(inds.shape)  #
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]
                gt_index = np.where(np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]
                label_matrix[index][gt_index] = 1
                caption2index.append((txt_ids[index], txt_loader.dataset.captions[txt_ids[index]],
                                   gt_index[0], tuple(np.array(vis_ids)[ind[0:10]])))
            # caption2index = sorted(caption2index, key=lambda kv: kv[2], reverse=True)  # 倒序排列
            (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
            sum_recall = r1 + r5 + r10

            result_file_dir = os.path.dirname(opt.predict_result_file)
            result_file_name = os.path.basename(opt.predict_result_file)

            write_to_predict_result_file(
                os.path.join(result_file_dir, 'TextToVideo', result_file_name),
                opt.model_path+"\t"+testCollection, checkpoint,
                (r1, r5, r10, medr, meanr, mir, mAP)
            )
            txt2video_write_to_file(None, inds, vis_ids, txt_ids, t2i_matrix,
                                    pkl_saved_file=os.path.join(output_dir, "t2v.pkl"), txt_loader=txt_loader,
                                    Threshold=500)

            # Video to Text
            i2t_matrix = t2i_matrix.T
            inds = np.argsort(i2t_matrix, axis=1)
            label_matrix = np.zeros(inds.shape)
            txt_ids = [txt_id.split('#')[0] for txt_id in txt_ids]
            for index in range(inds.shape[0]):
                ind = inds[index][::-1]
                label_matrix[index][np.where(
                    np.array(txt_ids)[ind] == vis_ids[index])[0]] = 1
            (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
            write_to_predict_result_file(
                os.path.join(result_file_dir, 'VideoToText', result_file_name),
                opt.model_path+"\t"+testCollection, checkpoint,
                (r1, r5, r10, medr, meanr, mir, mAP),
                name_str="Video To Text"
            )

            continue

        txt2video_write_to_file(None, inds, vis_ids, txt_ids, t2i_matrix,
                                pkl_saved_file=os.path.join(output_dir, "t2v.pkl"), txt_loader=txt_loader,
                                Threshold=500)
        start = time.time()
        txt2video_write_to_file(pred_result_file, inds, vis_ids, txt_ids, t2i_matrix)

        print('writing to %s\n' % (pred_result_file))
        print('writing result into file time: %.3f seconds\n' % (time.time() - start))


def get_multi_predict_file(opt, checkpoint):
    """
    把每个头的结果取出来
    :param opt:
    :param checkpoint:
    :return:
    """
    rootpath = opt.rootpath
    testCollection = opt.testCollection
    task3_caption_suffix = opt.task3_caption
    # cuda number
    device = torch.device("cuda:{}".format(opt.device)
                          if (torch.cuda.is_available() and opt.device != "cpu") else "cpu")

    resume_file = os.path.join(opt.model_path)

    # Load checkpoint
    epoch = checkpoint['epoch']
    best_perf = checkpoint['best_perf']
    config = checkpoint['config']
    model_name = checkpoint['config'].model_name
    if opt.task3_caption == "no_task3_caption":
        task3 = False
    else:
        task3 = True
    if hasattr(config, 't2v_w2v') and hasattr(config.t2v_w2v, 'w2v'):
        w2v_feature_file = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m', 'feature.bin')
        config.t2v_w2v.w2v.binary_file = w2v_feature_file

    # Construct the model
    model = get_model(model_name, device, config)
    model = model.to(device)
    # print(model)
    # calculate the number of parameters
    vis_net_params = sum(p.numel() for p in model.vis_net.parameters())
    txt_net_params = sum(p.numel() for p in model.txt_net.parameters())
    print('    VisNet params: %.2fM' % (vis_net_params / 1000000.0))
    print('    TxtNet params: %.2fM' % (txt_net_params / 1000000.0))
    print('    Total params: %.2fM' %
          ((vis_net_params + txt_net_params) / 1000000.0))

    model.load_state_dict(checkpoint['model'], strict=False)
    print("=> loaded checkpoint '{}' (epoch {}, best_perf {})"
          .format(resume_file, epoch, best_perf))

    vis_feat_files = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData', y))
                      for y in config.vid_feats}
    # 视频帧特征文件
    vis_frame_feat_dicts = None
    if config.frame_feat_input:
        vis_frame_feat_dicts = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData/frame', y))
                                for y in config.vid_frame_feats}
    vis_ids = list(map(str.strip, open(os.path.join(rootpath, testCollection, 'VideoSets', testCollection + '.txt'))))
    vis_loader = data.vis_provider({'vis_feat_files': vis_feat_files, 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts,
                                    'max_frame': config.max_frame,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

    for query_set in opt.query_sets.split(','):
        output_dir = os.path.join(rootpath, testCollection, 'SimilarityIndex', query_set, opt.sim_name)
        pred_result_file = os.path.join(output_dir, 'id.sent.score.txt')

        if util.checkToSkip(pred_result_file, opt.overwrite):
            continue
        util.makedirs(output_dir)

        capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
        # load text data
        txt_loader = data.txt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                        'batch_size': opt.batch_size, 'num_workers': opt.num_workers, 'task3': task3})

        t2i_matrixs, txt_ids, vis_ids = model.predict_each_head(txt_loader, vis_loader, measure=config.measure)
        for i in range(t2i_matrixs.shape[0]):
            t2i_matrix = t2i_matrixs[i, :, :]
            inds = np.argsort(t2i_matrix, axis=1)

            if testCollection in ['msrvtt10ktest', 'msrvtt1kAtest', 'tgiftest', 'msvdtest',
                                  'vatex_pub_test'] and query_set != "simple_query.txt":
                # caption2index 里面是 ('video001#1', caption, 1, [video001, ...])，这样的 caption 到 gt 检索结果的形式，最后是前10个结果。
                caption2index = []
                label_matrix = np.zeros(inds.shape)  #
                for index in range(inds.shape[0]):
                    ind = inds[index][::-1]
                    gt_index = np.where(np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]
                    label_matrix[index][gt_index] = 1
                    caption2index.append((txt_ids[index], txt_loader.dataset.captions[txt_ids[index]],
                                          gt_index[0], tuple(np.array(vis_ids)[ind[0:10]])))
                # caption2index = sorted(caption2index, key=lambda kv: kv[2], reverse=True)  # 倒序排列

                (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
                sum_recall = r1 + r5 + r10
                tempStr = " * Text to video head" + str(i) + ":\n"
                tempStr += " * r_1_5_10: {}\n".format([round(r1, 3), round(r5, 3), round(r10, 3)])
                tempStr += " * medr, meanr, mir: {}\n".format([round(medr, 3), round(meanr, 3), round(mir, 3)])
                tempStr += " * mAP: {}\n".format(round(mAP, 3))
                tempStr += " * " + '-' * 10
                print(tempStr)
                open(os.path.join(output_dir, 'perf.txt'), 'w').write(tempStr)

                with open(opt.reult_file, 'a') as f:
                    f.write(str(time.asctime(time.localtime(time.time()))) + '\t')
                    for each in [opt.model_path, i, round(r1, 3), round(r5, 3), round(r10, 3),
                                 round(medr, 3), round(meanr, 3), round(mir, 3), round(mAP, 3)]:
                        f.write(str(each))
                        f.write('\t')
                    f.write(checkpoint['opt'].parm_adjust_config.replace('_', '\t'))
                    f.write('\n')

            start = time.time()
            with open(pred_result_file, 'w') as fout:
                for index in range(inds.shape[0]):
                    ind = inds[index][::-1]

                    fout.write(txt_ids[index] + ' ' + ' '.join([vis_ids[i] + ' %s' % t2i_matrix[index][i]
                                                                for i in ind]) + '\n')
            print('writing result into file time: %.3f seconds\n' % (time.time() - start))



def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    if '~' in opt.rootpath:
        opt.rootpath = opt.rootpath.replace('~', os.path.expanduser('~'))
    # Load checkpoint
    logger.info('loading model...')
    resume_file = os.path.join(opt.model_path)
    if '~' in resume_file:
        resume_file = resume_file.replace('~', os.path.expanduser('~'))
        opt.model_path = resume_file
    if not os.path.exists(resume_file):
        logging.info(resume_file + '\n not exists.')
        sys.exit(0)
    checkpoint = torch.load(resume_file, map_location='cpu')
    # set the config parm you adjust

    # if checkpoint['opt'].parm_adjust_config != 'None':
    #     checkpoint['config'].adjust_parm(checkpoint['opt'].parm_adjust_config)
    checkpoint['opt'].device = "cpu"
    checkpoint['opt'].rootpath = opt.rootpath
    # checkpoint['opt'].pretrained_file_path = 'None'
    prepared_configs = trainer.prepare_config(checkpoint['opt'], False)
    config = prepared_configs['config']
    del prepared_configs
    checkpoint['config'] = config

    get_predict_file(opt, checkpoint)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print()
        # LAFF
        sys.argv = "predictor.py --device 1 msrvtt1kAtest " \
                   "~/VisualSearch/msrvtt10ktrain/w2vvpp_train/msrvtt10kval/CVPR.FrameLaff_NoFrameFc_StrongCLIP_adjust/runs_0_7_1_12_0_12_0_seed_2/model_best.pth.tar " \
                   "msrvtt10ktrain/msrvtt10kval/laff " \
                   "--rootpath ~/VisualSearch --batch_size 512 " \
                   "--query_sets simple_query.txt " \
                   "--overwrite 1".split(' ')


    main()
