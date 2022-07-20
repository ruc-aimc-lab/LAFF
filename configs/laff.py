# sea_avs_adjustVisTxt
from . import base_config as BaseConfig
import numpy as np


class config(BaseConfig.config):
    model_name = 'LAFF'  # 选择的模型，见 model.py
    # visual Attention
    dropout = 0.2
    activation = 'tanh'
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
    # if text_encoding includes bert
    bert_size = 768
    bert_frozen = True
    bert_do_lower_case = True
    bert_transform_batch_norm = True
    bert_transform_dropout = 0
    bert_transform_activation = 'tanh'

    # if text_encoding includes CLIP
    clip_opt = {
        'size': 512, 'transform_batch_norm': True, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': True, 'vocab_size': 49408,
    }
    # For Multi head
    attention_param_each_head = {
        "with_ave": True, "mul": False, 'split_head': True,
    }
    multi_head_attention = {  # if attention include muti_head_attention
        'dropout': 0.0, 'heads': 8, 'embed_dim_qkv': 4096 // 8,

    }
    vis_attention_global_decay_rate = 0.8
    txt_attention_global_decay_rate = 0.8
    vis_no_transform = ['clip_finetune_8frame_uniform_1103']  # ['clip_finetune_8frame_uniform_1103']#[ 'clip_finetune_0922']
    txt_no_transform = ['CLIP_encoder']  # ['CLIP_encoder']

    # For Attention params
    def adjust_parm(self, value):
        vid_feats = ['clip_finetune_8frame_uniform_1103', 'mean_resnext101_resnet152',
                     'mean_C3d_resneXt101_16f', 'mean_resnext101_32x48d_wsl,avgpool,os',
                     'mean_pyresnext-101_rbps13k,flatten0_output,os', 'HowTo100M_TimeSformer_divST_96x4_224',
                     'X3D_L',  'mean_irCSN_152_ig65m_from_scratch',
                     ]
        # 0:clip, 1:res101+152, 2:c3d,
        # 3:wsl, 4:101, 5:timesformer
        # 6:X3D, 7 ircsn
        # 10: 152
        vid_feats_iterlist = [
            np.array([0, 5, 6, 7]),  # 0 clip+timesformer+x3d+ircsn
        ]
        text_encodings = [
            ['bow_nsw', 'w2v_nsw', 'gru_mean', 'noBert', 'ViT-B/32', 'noNetVLAD'],
            # 0 (bow+w2v+gru+clip)
        ]
        a = []
        for i, each in enumerate(value.split('_')):
            a.append(eval(each))
        print(a)
        self.vid_feats = list(np.array(vid_feats)[vid_feats_iterlist[a[0]]])  # 7 8
        print("vid_feats", self.vid_feats)
        self.vis_attention = self.vis_attentions[a[1]]
        #  8(concat) 9(attention) 11(multi_head + official_self-attention)
        # 12(LAFF) attention_types

        # text_encoding
        for i, each in enumerate(self.text_encoding):
            self.text_encoding[each]['name'] = text_encodings[a[2]][i]

        self.txt_attention = self.txt_attentions[a[3]]

        self.attention_param_each_head["with_ave"] = True if a[4] == 1 else False
        self.attention_param_each_head["mul"] = True if a[5] == 1 else False
        self.attention_param_each_head["split_head"] = True if a[6] == 1 else False

