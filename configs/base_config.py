# coding=utf-8

class config(object):
    def adjust_parm(self, value):
        pass

    def get_txt_encoder_num(self, text_encoding):
        encoder_num = 0
        for name in text_encoding:
            encoder_value = text_encoding[name]['name']
            if 'no' not in encoder_value:
                encoder_num += 1

        return encoder_num

    model_name = 'w2vpp_mutivis_attention'  # 选择的模型，见 model.py

    text_encoding = {
        'bow_encoding': {'name': 'bow_nsw'},  # [nobow_nsw, bow_nsw]
        'w2v_encoding': {'name': 'w2v_nsw'},  # [now2v_nsw, w2v_nsw]
        'rnn_encoding': {'name': 'gru_mean'},  # [gru_mean, bigru_mean, nogru_mean]
        'bert_encoding': {'name': 'noBert',  # [noBert, bert-base-uncased, \bert_name, ...]
                          'dir_name': 'bert-base-uncased'
                          },
        'CLIP_encoding': {'name': 'noCLIP',  # [noCLIP, ViT-B/32, \CLIP_name, ...]
                          'dir_name': 'CLIP_ViT-B32'
                          },
        'NetVLAD_encoding': {'name': 'noNetVLAD'},  # [noNetVLAD, NetVLAD]
        # 'SLIP_encoding': {'model_path': 'noSLIP',  # [noCLIP, ViT-B/32, \CLIP_name, ...]
        #                   'dir_name': 'SLIP_ViT-B32'
        #                   },
    }
    preprocess_type = 'clip'
    text_encoder_num = 3
    threshold = 5
    bow_norm = 0
    we_dim = 500
    rnn_size = 1024
    rnn_layer = 1
    txt_fc_layers = '0-2048'
    txt_norm = 2  # L_2 norm


    # txt encoder and transform
#
    # if text_encoding includes bert
    bert_size = 768
    bert_frozen = False
    bert_do_lower_case = True
    bert_transform_batch_norm = True
    bert_transform_dropout = 0
    bert_transform_activation = 'tanh'
    # if text_encoding includes CLIP
    clip_opt = {
        'size': 512, 'transform_batch_norm': False, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': True, 'vocab_size': 49408,
    }
    # if text_encoding includes SLIP
    slip_opt = {
        'size': 512, 'transform_batch_norm': False, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': True
    }
    # if text_encoding includes NetVLAD
    NetVLAD_opt = {
        'num_clusters': 32, 'alpha': 100, 'normalize_pooling': False,
    }

    # visual transform
    vis_fc_layers = ['0', 2048]
    vis_norm = 2  # L_2 norm
    use_abs = False
    batch_norm = False
    batch_norm_momentum = 0.1
    batch_norm_eps = 1e-05
    # dropout
    dropout = 0.2  # 0.4 is better
    last_dropout = 0.2
    # activation
    activation = 'tanh'
    last_activation = 'tanh'


    # loss
    loss = 'mrl'  # [dsl]
    margin = 0.2
    direction = 't2i'  # ['t2i', 'bidir']. only valid for mrl loss
    # Use max instead of sum in the rank loss
    max_violation = True  # only valid for mrl loss
    # cost style (sum|mean)
    cost_style = 'sum'  # only valid for mrl loss
    # Similarity measure used (cosine|order)
    measure = 'cosine'

    # optimizer
    optimizer = 'rmsprop'
    # Initial learning rate.
    lr = 0.0001
    lr_decay_rate = 0.99
    # Gradient clipping threshold
    grad_clip = 2

    # half model: float16 tensor, half the memory. Recommend to True.
    float16 = False

    # ********************************萌萌哒分界线******************
    # Attention
    attention_types = ('attention_noAverageMul_Ave',  # 0 attention_noAverageMul_Ave: 添加 meanpooling，不进行 ave_global 与 local 相乘
                      'average_AverageMul_noAve',  # 1 average_attention: 不添加 meanpooling，进行 ave_global 与 local 相乘
                      'con_attention',
                      'fc_attention',
                       'just_average',  # 4
                       'muti_head_attention',
                      'attention3',
                      'attention_noAveNoAverageMul',  # 7 attention_noAveNoAverageMul: 不添加 meanpooling，不进行 ave_global 与 local 相乘
                      'concat',  # 8 concat: 如 w2vvpp 那样拼接特征
                      'attention_averageMul',  # 9 进行 ave_global 与 local 相乘
                      'muti_head_attention_official',  # 10 官方 multi-head
                      'my_self_attention',  # 11 我的 multi_head + official_self-attention，返回多个 head 结果
                      'Multi_head_MyApply_Attention',  # 12 我的 multi_head + attention，返回多个 head 结果
                      'Multi_head_MyApply_FusionAttention',  # 13 我的 multi_head + 多个 attention，返回多个 head 结果
                      'Multi_head_Attention_layer_norm',  # 14 我的 multi_head + layer_norm，返回多个 head 结果
                      'Multi_head_Attention_distinct_fc',  # 15 我的 multi_head + l2norm_distinct_fc，返回多个 head 结果
                      'Attention_MMT',  # 16 Attention_MMT
                      )
    attention_l2norm = False
    muti_head_attention_official = {'agg': 'mean'}
    vis_attentions = attention_types

    vis_no_transform = []  # ['clip_finetune_8frame_uniform_1103']#[ 'clip_finetune_0922']
    txt_no_transform = []  # ['CLIP_encoder']


    my_self_attention_output_types = ['mean', 'max', 'first', 'last', 'cls_embedding',
                                      'concat', 'max_embedding', 'mean_embedding', 'random', 'second',
                                      'third', 'Attention_1']
    my_self_attention_output_type = my_self_attention_output_types[0]


    # Txt Attention
    txt_attentions = attention_types
    txt_attention = attention_types[1]

    txt_attention_global_decay_rate = 0.8  # 0.8 衰减
    txt_expert_embedding = {'expert': False, 'l2norm': False}

    # visual Attention
    vid_feats = ['mean_resnext101_resnet152', 'irCSN_152_ig65m_16frms',
                 'mean_pyresnext-101_rbps13k,flatten0_output,os', 'ipcsn_sports1m_32frms',
                 'mean_C3d_resneXt101_16f', 'mean_resnext101_32x48d_wsl,avgpool,os',
                 # 21.5.7
                 'mean_clip_frame_feat_ViT-B_32,os', 'HowTo100M_TimeSformer_divST_96x4_224',
                 'X3D_L', 'I3D_NLN_8x8_R50',
                 ]
    vis_feat_add_concat = False  # 是否增加一个所有特征拼接特征

    vis_attention = attention_types[1]
    vis_attention_global_decay_rate = 0.8  # 0.8 不衰减
    vis_expert_embedding = {'expert': False, 'l2norm': False}

    multi_head_attention = {  # if attention include muti_head_attention
        'dropout': 0.0, 'heads': 4, 'embed_dim_qkv': 2048 // 4,

    }
    attention_param_each_head = {
        "with_ave": True, "mul": False, 'split_head': True,
    }
    multi_space = True  # 每个 head 一个 space，一个 loss，或者把所有 sapce 的结果相加

    # visual frame feats
    max_frame = 200  # 最大输入 max_frame 帧
    frame_feat_input = False
    frame_feat_with_video_feat = False  # 是否加上 video_feats，if model == w2vpp_MutiVisFrameFeat_attention
    vid_frame_feats = ['pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os',
                       ]
    vis_frame_attention = attention_types[1]
    vis_frame_addFC = True


    # ********************************萌萌哒分界线******************
    # MMT https://github.com/gabeur/mmt
    tranformer_encoder_opt = {
        'nhead': 4, 'num_layers': 4
    }
    add_vid_feats = False  # 是否加上 video_feats

    # ********************************萌萌哒分界线******************
    # SEA, if model='w2vpp_sea'

    # SEA, if model='w2vvpp_sea_weight'
    # txt_encoder_lists = ('list', 'list_transform_to_subspace')  #
    # txt_encoder_list = txt_encoder_lists[0]
    # score_attentions = ('average_attention', 'fc_attention', 'just_average', 'use_weights')
    # score_attention = score_attentions[2]
    # sea_trainstyles = ('train_together', 'train_split')  # 'train_split'：分开训练。线性层梯度不回传
    # sea_trainstyle = sea_trainstyles[0]
    # score_attention_params = ('usual_learning_rate', '10times')
    # score_attention_param = score_attention_params[0]

    # ********************************萌萌哒分界线******************
    # ircsn video process
    csn = False

    # ********************************萌萌哒分界线******************
    # SGRAF model setting
    SGRAF = False
    muti_feat = 'vg_label_feat_36dim_repeat'  # ['vg_label_feat_36dim_repeat', 'vg_label_feat']
    img_dim = 2048
    word_dim = 300
    embed_size = 1024
    sim_dim = 256
    num_layers = 1  # Number of GRU layers.
    bi_gru = False
    no_imgnorm = True
    no_txtnorm = True
    module_name = 'SGR'  # SGR, SAF
    sgr_step = 3  # step of SGR

    # ********************************萌萌哒分界线******************
    task2 = False
    # task2 text embbeding
    txt_feature_task2 = 'bow'  # 'bow, gru, w2v, no'
    txt_fc_layers_task2 = '0-0'  # 根据txt_feature_task2自动调整
    # task2 multi label
    text_encoding_task2 = 'bow_nsw'
    threshold_task2 = 5  # 词需要出现5次以上
    bow_norm_task2 = 0  # 对 word -> vec 求范数
    batch_norm_task2 = True
    activation_task2 = 'sigmoid'  # no, tanh, relu, sigmoid
    dropout_task2 = 0.1
    # task2 visual embbeding
    vis_fc_layers_task2 = '0-0'  # 第一个设置为vis_fc_layers输入(在trainer中调整)

    # ********************************萌萌哒分界线******************
    # Negative
    task3_start=-1
    task3_loss_weight=1
    task3_margin = 0.2
    # loss
    loss_lambda = 0.2
    # Similarity measure used (cosine|order|hist)  # hist is jaccard sim
    measure_task2 = 'hist'
    # parameter that balance latent space and task2 space (concept space)
    alpha = 0.2

    negative = False
    kl = False
    mask = False
    origin_vid_feats = None
    origin_text_feats = None

    task3_end=100
    task3_neg_weight = 1
    task3_neg_retrival_weight = 0.001
    task3_bottommargin = 0.1
    task3_uppermargin = 0.6
    task3_bottommargin_t2t = 0.1
    task3_uppermargin_t2t = 0.3
    max_txtlength = 77

    # ********************************萌萌哒分界线******************
    # end2end 学习，输入 frame/video 原始文件
    frame_loader = False
    frame_sample_type_train = 'random'  # ['uniform', 'random']
    frame_sample_type_test = 'uniform'
    sample_frame = 8  # 每个视频均匀选 sample_frame 张

    # Feature re-learning
    txt_fc_same_with_vis_fc = False  # txt fc 和 vis fc 相同
    txt_fc_same_with_vis_fc_dict = {
        'CLIP_encoder': 'clip2video_global_visual_output_MSVD',  # 'CLIP_encoding' FC 与视频特征 clip2video_global_visual_output_MSVD FC 相同
    }
    skip_feature = {
        'visual': None,
        'text': None,
    }
    # ********************************萌萌哒分界线******************

