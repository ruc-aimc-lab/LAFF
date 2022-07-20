from . import base_config as BaseConfig
import numpy as np


class config(BaseConfig.config):
    model_name = 'FrameLAFF'  # 选择的模型，见 model.py
    # visual Attention
    dropout = 0.2
    activation = 'tanh'
    batch_norm = True
    vis_fc_layers = ['0', 4096]
    txt_fc_layers = '0-4096'

    # txt encoder and transform
    text_encoding = {
        'bow_encoding': {'name': 'bow_nsw'},  # [nobow_nsw, bow_nsw]
        'w2v_encoding': {'name': 'w2v_nsw'},  # [now2v_nsw, w2v_nsw]
        'rnn_encoding': {'name': 'gru_mean'},  # [gru_mean, bigru_mean, nogru_mean]
        'bert_encoding': {'name': 'noBert',  # [noBert, bert-base-uncased, \bert_name, ...]
                          'dir_name': 'bert-base-uncased'
                          },
        'CLIP_encoding': {'name': 'noCLIP',  # [noCLIP, ViT-B/32, \CLIP_name, ...]
                          'dir_name': 'clip_finetune_8frame_uniform_1103'
                          },
        'NetVLAD_encoding': {'name': 'noNetVLAD'},  # [noNetVLAD, NetVLAD]
    }

    # if text_encoding includes CLIP
    clip_opt = {
        'size': 512, 'transform_batch_norm': True, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': True
    }
    float16 = True

    max_frame = 50  # 最大输入 max_frame 帧
    frame_feat_input = True
    vid_frame_feats = ['clip_frame_feat_ViT-B_32,os',
                       ]

    vis_frame_attention = BaseConfig.config.attention_types[1]

    # For Multi head
    attention_param_each_head = {
        "with_ave": False, "mul": False, 'split_head': True,
    }
    multi_head_attention = {  # if attention include muti_head_attention
        'dropout': 0.0, 'heads': 8, 'embed_dim_qkv': 4096 // 8,

    }
    vid_feats = ['mean_clip_frame_feat_ViT-B_32,os']
    frame_feat_with_video_feat = True

    vis_attention_global_decay_rate = 0.0
    txt_attention_global_decay_rate = 0.0

    vis_no_transform = ['clip_finetune_8frame_uniform_1103', 'clip_frame_feat_ViT-B_32,os']  # ['clip_finetune_8frame_uniform_1103']#[ 'clip_finetune_0922']
    txt_no_transform = ['CLIP_encoder']  # ['CLIP_encoder']
    vis_frame_addFC = False

    # For Attention params
    def adjust_parm(self, value):
        vid_frame_feats = [
            'Frame_clip_finetune_8frame_uniform_1103',
            'clip_frame_feat_ViT-B_32,os',
                           ]
        # 0:Frame_clip_finetune, 1:clip_frame
        clip_precal_feats = ['clip_finetune_8frame_uniform_1103', 'CLIP_ViT-B32']

        vid_feats_iterlist = [
            np.array([0]),  # 0 clip-ft
            np.array([1]),  # 1 clip
        ]
        text_encodings = [
            # 0 clip
            ['nobow_nsw', 'now2v_nsw', 'nogru_mean', 'noBert', 'ViT-B/32', 'noNetVLAD'],
            # 1 bow+w2v+gru+clip
            ['bow_nsw', 'w2v_nsw', 'gru_mean', 'noBert', 'ViT-B/32', 'noNetVLAD'],
            # 2 bow+w2v+clip
            ['bow_nsw', 'w2v_nsw', 'nogru_mean', 'noBert', 'ViT-B/32', 'noNetVLAD'],
        ]


        a = []
        for i, each in enumerate(value.split('_')):
            a.append(eval(each))

        self.vid_frame_feats = list(np.array(vid_frame_feats)[vid_feats_iterlist[a[0]]])  # 0
        self.vis_no_transform = list(np.array(vid_frame_feats)[vid_feats_iterlist[a[0]]])
        print("vid_frame_feats", self.vid_frame_feats)
        self.text_encoding['CLIP_encoding']['dir_name'] = clip_precal_feats[a[0]]

        self.vis_frame_attention = self.attention_types[a[1]]  # 12 7(attention)

        # text_encoding
        for i, each in enumerate(self.text_encoding):
            self.text_encoding[each]['name'] = text_encodings[a[2]][
                i]  # 0(clip) 1(bow+w2v+gru+clip) 2(bow+w2v+clip)

        self.txt_attention = self.txt_attentions[a[3]]  # 12

        # vis_attention
        vid_feats = ['mean_clip_frame_feat_ViT-B_32,os', 'mean_resnext101_resnet152',
                     'mean_C3d_resneXt101_16f', 'mean_resnext101_32x48d_wsl,avgpool,os',
                     'mean_pyresnext-101_rbps13k,flatten0_output,os', 'HowTo100M_TimeSformer_divST_96x4_224',
                     'X3D_L',  'mean_irCSN_152_ig65m_from_scratch',
                     'random_feat_512', 'full_1_feat_512',
                     'mean_pyresnet-152_imagenet11k,flatten0_output,os',

                     ]
        # 0:clip, 1:res101+152, 2:c3d,
        # 3:wsl, 4:101, 5:timesformer
        # 6:X3D
        vid_feats_iterlist = [
            np.array([2, 5, 6, 7]),  # 0 timesformer+x3d+ircsn+c3d
            np.array([4, 2, 3, 7]),  # 1 101+c3d+wsl+ircsn

        ]
        self.vid_feats = list(np.array(vid_feats)[vid_feats_iterlist[a[4]]])
        print("vid_feats", self.vid_feats)
        self.vis_attention = self.attention_types[a[5]]  # 12



