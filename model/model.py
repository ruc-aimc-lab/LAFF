# coding=utf-8

import torch
import sys

sys.path.append('../')
import model.clip as clip
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import BertTokenizer, BertModel
# import vmz.models as models

import evaluation
from . import ReRank
import util
from loss import *
from loss import l2norm
from bigfile import BigFile
from common import logger
from generic_utils import Progbar
from model.Attention import *



def get_we(vocab, w2v_dir):
    """
    得到 word2vec 模型，n x 500
    :param vocab:
    :param w2v_dir:
    :return: word2vec 参数，[11286, 500]
    """
    w2v = BigFile(w2v_dir)
    ndims = w2v.ndims
    nr_words = len(vocab)
    words = [vocab[i] for i in range(nr_words)]
    we = np.random.uniform(low=-1.0, high=1.0, size=(nr_words, ndims))

    renamed, vecs = w2v.read(words)
    for i, word in enumerate(renamed):
        idx = vocab.find(word)
        we[idx] = vecs[i]

    return torch.Tensor(we)


def _initialize_weights(m):
    """Initialize module weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm1d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def to_device_and_float16(x: torch.Tensor):
    x = x.to(device)
    # if float16:
    #     x = x.half()
    return x


def get_attention_layer(attention_type: str, common_space_dim, encoder_num, opt):
    """
    :parm attention_type: 选择的 attention
    :param common_space_dim:
    :param encoder_num:
    :return:
    """
    def cal_params():
        common_space_dim = 2048
        heads = 8
        dim_per_head = common_space_dim // heads
        split_head = True
        net = Multi_head_MyApply_Attention(
            common_space_dim, heads,
            dim_per_head,
            split_head=split_head,
        )
        net = Attention_multi_head_official(
                            common_space_dim,
                            heads)
        net = nn.Linear(2048, 1)
        net_params = sum(p.numel() for p in net.parameters())
        print('Net params: %.8fM' % (net_params / 1000000.0))
        pass
    try:
        attention_layers = {'attention_noAverageMul_Ave': Attention_1(common_space_dim, with_ave=True, mul=False),
                            'attention_noAveNoAverageMul': Attention_1(common_space_dim, with_ave=False, mul=False),
                            'attention_averageMul': Attention_1(common_space_dim, with_ave=True, mul=True),
                            'average_AverageMul_noAve': Attention_1(common_space_dim, with_ave=False, mul=True),
                            'con_attention': nn.Sequential(nn.Conv1d(encoder_num, 1, 1)),
                            'fc_attention': FcAttention(encoder_num),
                            'just_average': JustAverage(),
                            'muti_head_attention': Attention_2(common_space_dim, opt),
                            'attention3': Attention_3(common_space_dim),
                            'muti_head_attention_official': Attention_multi_head_official(
                                common_space_dim,
                                8, opt.multi_head_attention['dropout'],
                                opt.muti_head_attention_official['agg']
                            ),
                            'Attention_MMT': Attention_MMT(
                                common_space_dim,
                                8, opt.multi_head_attention['dropout']),
                            'my_self_attention': Multi_head_MyApply_selfAttention(
                                common_space_dim, opt.multi_head_attention['heads'],
                                # opt.multi_head_attention['embed_dim_qkv'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                opt.multi_head_attention['dropout'],
                                output_type=opt.my_self_attention_output_type,
                                encoder_num=encoder_num,
                                l2norm_each_head=opt.attention_l2norm,
                                opt=opt
                            ),
                            'Multi_head_MyApply_Attention': Multi_head_MyApply_Attention(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                with_ave=opt.attention_param_each_head['with_ave'],
                                mul=opt.attention_param_each_head['mul'],
                                split_head=opt.attention_param_each_head['split_head'],
                                l2norm_each_head=opt.attention_l2norm,
                            ),
                            'Multi_head_MyApply_FusionAttention': Multi_head_MyApply_FusionAttention(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                opt.attention_param_each_head['split_head'],
                            ),
                            'Multi_head_Attention_distinct_fc': Multi_head_Attention_distinct_fc(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                with_ave=opt.attention_param_each_head['with_ave'],
                                mul=opt.attention_param_each_head['mul'],
                                split_head=opt.attention_param_each_head['split_head'],
                            ),
                            'Multi_head_Attention_layer_norm': Multi_head_Attention_layer_norm(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                with_ave=opt.attention_param_each_head['with_ave'],
                                mul=opt.attention_param_each_head['mul'],
                                split_head=opt.attention_param_each_head['split_head'],
                            ),
                            }
    except Exception as e:
        print(e)
    attention_layers = {'attention_noAverageMul_Ave': Attention_1(common_space_dim, with_ave=True, mul=False),
                        'attention_noAveNoAverageMul': Attention_1(common_space_dim, with_ave=False, mul=False),
                        'attention_averageMul': Attention_1(common_space_dim, with_ave=True, mul=True),
                        'average_AverageMul_noAve': Attention_1(common_space_dim, with_ave=False, mul=True),
                        'con_attention': nn.Sequential(nn.Conv1d(encoder_num, 1, 1)),
                        'fc_attention': FcAttention(encoder_num),
                        'just_average': JustAverage(),
                        'muti_head_attention': Attention_2(common_space_dim, opt),
                        'attention3': Attention_3(common_space_dim),
                        'muti_head_attention_official': Attention_multi_head_official(
                            common_space_dim,
                            8, opt.multi_head_attention['dropout'],
                            opt.muti_head_attention_official['agg']
                        ),
                        'Attention_MMT': Attention_MMT(
                            common_space_dim,
                            8, opt.multi_head_attention['dropout']),
                        'my_self_attention': Multi_head_MyApply_selfAttention(
                            common_space_dim, opt.multi_head_attention['heads'],
                            # opt.multi_head_attention['embed_dim_qkv'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            opt.multi_head_attention['dropout'],
                            output_type=opt.my_self_attention_output_type,
                            encoder_num=encoder_num,
                            l2norm_each_head=opt.attention_l2norm,
                            opt=opt
                        ),
                        'Multi_head_MyApply_Attention': Multi_head_MyApply_Attention(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            with_ave=opt.attention_param_each_head['with_ave'],
                            mul=opt.attention_param_each_head['mul'],
                            split_head=opt.attention_param_each_head['split_head'],
                            l2norm_each_head=opt.attention_l2norm,
                        ),
                        'Multi_head_MyApply_FusionAttention': Multi_head_MyApply_FusionAttention(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            opt.attention_param_each_head['split_head'],
                        ),
                        'Multi_head_Attention_distinct_fc': Multi_head_Attention_distinct_fc(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            with_ave=opt.attention_param_each_head['with_ave'],
                            mul=opt.attention_param_each_head['mul'],
                            split_head=opt.attention_param_each_head['split_head'],
                        ),
                        'Multi_head_Attention_layer_norm': Multi_head_Attention_layer_norm(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            with_ave=opt.attention_param_each_head['with_ave'],
                            mul=opt.attention_param_each_head['mul'],
                            split_head=opt.attention_param_each_head['split_head'],
                        ),
                        }

    return attention_layers[attention_type]


class TransformNet(nn.Module):
    """
    fc_layers: (dim_in, dim_out)
    加入 BatchNorm, activation, dropout
    """

    def __init__(self, fc_layers, opt=None, dropout=None, batch_norm=None, activation=None, fc=True):
        super(TransformNet, self).__init__()

        if opt is not None:
            if batch_norm is None:
                batch_norm = opt.batch_norm
            if activation is None:
                activation = opt.activation
            if dropout is None:
                dropout = opt.dropout
        if fc:
            self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        else:
            self.fc1 = None
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(fc_layers[1])
        else:
            self.bn1 = None

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

        if dropout is not None and dropout > 1e-3:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.apply(_initialize_weights)

    def forward(self, input_x):
        """
        一般来说顺序：-> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        不过有了 bn 一般不用 dropout
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """
        features = input_x.to(device)
        if self.fc1 is not None:
            features = self.fc1(features)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        if self.bn1 is not None:
            features = self.bn1(features)

        return features


class VisTransformNet(TransformNet):
    """
    把拼接的 video_emb 映射到公共空间
    """

    def __init__(self, opt):
        super(VisTransformNet, self).__init__((np.sum(list(opt.vis_fc_layers[0].values())), opt.vis_fc_layers[1]), opt)

    def forward(self, vis_input: dict, txt_emb=None, vis_frame_feat_dict_input=None):
        """
        一般来说顺序：-> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        不过有了 bn 一般不用 dropout
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """
        if type(vis_input) == dict:
            vis_feature = to_device_and_float16(torch.cat(list(vis_input.values()), dim=1))
        else:
            vis_feature = to_device_and_float16(vis_input)
        features = self.fc1(vis_feature)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        if self.bn1 is not None:
            features = self.bn1(features)

        return features


class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super(TxtEncoder, self).__init__()

    def forward(self, caption_feat_dict, task3=False):
        output = {}
        output['text_features'] = caption_feat_dict['caption']

        return output


class GruTxtEncoder(TxtEncoder):
    def _init_rnn(self, opt):
        if opt.rnn_size == 0:
            return
        self.rnn = nn.GRU(int(opt.we_dim), int(opt.rnn_size), int(opt.rnn_layer), batch_first=True, bidirectional=False)

    def __init__(self, opt):
        super().__init__(opt)
        self.bigru = False
        self.pooling = opt.pooling
        self.rnn_size = opt.rnn_size
        self.t2v_idx = opt.t2v_idx
        self.we = nn.Embedding(len(self.t2v_idx.vocab), opt.we_dim)
        if opt.we_dim == 500:
            self.we.weight = nn.Parameter(opt.we)  # initialize with a pre-trained 500-dim w2v

        self._init_rnn(opt)

    def forward(self, caption_feat_dict, task3=False):
        txt_input = caption_feat_dict['caption']
        batch_size = len(txt_input)

        # caption encoding
        idx_vecs = [self.t2v_idx.encoding(caption) for caption in txt_input]
        lengths = [len(vec) for vec in idx_vecs]

        x = to_device_and_float16(torch.zeros(batch_size, max(lengths)).long())
        for i, vec in enumerate(idx_vecs):
            end = lengths[i]
            x[i, :end] = torch.Tensor(vec)

        # caption embedding
        x = self.we(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)

        if self.pooling == 'mean':
            # out = torch.zeros(batch_size, padded[0].shape[-1]).to(device)
            out = x.new_zeros((batch_size, padded[0].shape[-1])).to(device)
            for i, ln in enumerate(lengths):
                out[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.pooling == 'mean_last':
            # out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            out = torch.cat((out1, out2), dim=1)

        output = {}
        output['text_features'] = out

        return output


class BiGruTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim, opt.rnn_size, opt.rnn_layer, batch_first=True, bidirectional=True)

    def __init__(self, opt):
        super().__init__(opt)
        self.bigru = True


class BoWTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super(BoWTxtEncoder, self).__init__(opt)
        self.t2v_bow = opt.t2v_bow

    def forward(self, caption_feat_dict,task3=False):
        txt_input = caption_feat_dict['caption']
        t = np.empty((len(txt_input), self.t2v_bow.ndims), )
        for i, caption in enumerate(txt_input):
            t[i] = self.t2v_bow.encoding(caption)

        # bow_out = torch.Tensor([self.t2v_bow.encoding(caption) for caption in txt_input]).to(device)
        bow_out = to_device_and_float16(torch.Tensor(t))
        # print(bow_out.shape)
        output = {}
        output['text_features'] = bow_out

        return output


class W2VTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super(W2VTxtEncoder, self).__init__(opt)
        self.t2v_w2v = opt.t2v_w2v

    def forward(self, caption_feat_dict, task3=False):
        txt_input = caption_feat_dict['caption']
        t = np.empty((len(txt_input), self.t2v_w2v.ndims), )
        for i, caption in enumerate(txt_input):
            t[i] = self.t2v_w2v.encoding(caption)

        w2v_out = to_device_and_float16(torch.Tensor(t))
        output = {}
        output['text_features'] = w2v_out

        return output


class BertTxtEncoder(nn.Module):
    """
    Bert encoder
    """

    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.bert_name = opt.text_encoding['bert_encoding']['name']  # 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name, do_lower_case=opt.bert_do_lower_case)
        self.BertModel = BertModel.from_pretrained(self.bert_name)

    def forward(self, caption_feat_dict, task3=False):
        if 'bert_encoding' in caption_feat_dict and self.opt.bert_frozen:
            features = caption_feat_dict['bert_encoding']
        else:
            txt_input = caption_feat_dict['caption']
            encoded_input = self.tokenizer(txt_input, return_tensors='pt', padding=True, truncation=True)
            for each in encoded_input:
                encoded_input[each] = to_device_and_float16(encoded_input[each])
            if self.opt.bert_frozen:
                with torch.no_grad():
                    bert_output = self.BertModel(**encoded_input)
            else:
                bert_output = self.BertModel(**encoded_input)
            features = bert_output['pooler_output']
        output = {}
        output['text_features'] = features

        return output


class CLIPEncoder(nn.Module):
    """
    CLIP encoder.
    transform text and image into features.
    """

    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.Clip_name = opt.text_encoding['CLIP_encoding']['name']
        self.frozen = opt.clip_opt['frozen']
        self.dim = opt.clip_opt['size']
        self.tokenizer = clip.tokenize
        self.simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.ClipModel, self.preprocess = clip.load(self.Clip_name, device=device, jit=False)

    def forward(self, caption_feat_dict, vis_origin_frame_tuple=None, task3=False,
                frame_agg_method='mean'):
        """

        :param caption_feat_dict:
        :param vis_origin_frame_tuple: ([sample_frame, 3, 224, 224], ...)
        :param task3:
        :return: (batch_size, dim)
        """
        output = {}
        # For text encoding
        if caption_feat_dict is not None:
            if 'CLIP_encoding' in caption_feat_dict and self.frozen:
                text_features = caption_feat_dict['CLIP_encoding']
            else:
                txt_input = caption_feat_dict['caption']
                text = to_device_and_float16(self.tokenizer(txt_input))
                if self.frozen and (not task3):
                    with torch.no_grad():
                        text_features = self.ClipModel.encode_text(text)
                else:
                    text_features = self.ClipModel.encode_text(text)
            output['text_features'] = text_features

        # For visual encoding
        if vis_origin_frame_tuple is not None:
            batch_size = len(vis_origin_frame_tuple)
            origin_frames = to_device_and_float16(torch.cat(vis_origin_frame_tuple, dim=0))

            if self.frozen:
                with torch.no_grad():
                    frame_features = self.ClipModel.encode_image(origin_frames)
            else:
                frame_features = self.ClipModel.encode_image(origin_frames)
            frame_features = frame_features.reshape((batch_size, -1, self.dim))
            if frame_agg_method == 'mean':
                visual_features = torch.mean(frame_features, dim=1)
            else:
                raise Exception("frame_agg_method is not applied.")

            output['visual_features'] = visual_features

        return output

class NetVLADTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.t2v_w2v = opt.t2v_w2v
        self.netvlad = NetVLAD(opt.t2v_w2v.ndims,
                               opt.NetVLAD_opt['num_clusters'],
                               opt.NetVLAD_opt['alpha'],
                               )

    def forward(self, caption_feat_dict, task3=False):
        captions = caption_feat_dict['caption']

        w2v_out = []
        for caption in captions:
            x = to_device_and_float16(torch.Tensor(self.t2v_w2v.raw_encoding(caption)))
            w2v_out.append(x)

        netvlad_out = self.netvlad(w2v_out)
        out_dict = {}
        out_dict['text_features'] = netvlad_out
        return out_dict


class MultiScaleTxtEncoder(TxtEncoder):
    """
    多个 txt net concatenate 叠加输出

    """

    def init_txt_encoder(self, opt):
        self.bow_encoding, self.w2v_encoding, self.rnn_encoding, \
        self.bert_encoding, self.CLIP_encoding, self.NetVLAD_encoding = (
            opt.text_encoding['bow_encoding']['name'],
            opt.text_encoding['w2v_encoding']['name'],
            opt.text_encoding['rnn_encoding']['name'],
            opt.text_encoding['bert_encoding']['name'],
            opt.text_encoding['CLIP_encoding']['name'],
            opt.text_encoding['NetVLAD_encoding']['name'],
        )
        self.space_dict = {}  # encoder: space_dimension
        self.txt_encoder_num = 0
        self.encoder = nn.Module()

        # gru
        self.rnn_encoding, opt.pooling = self.rnn_encoding.split('_', 1)
        if self.rnn_encoding == 'gru':
            self.space_dict['rnn_encoder'] = opt.rnn_size
            self.txt_encoder_num += 1
            self.encoder.add_module('rnn_encoder', GruTxtEncoder(opt))
        elif self.rnn_encoding == 'bigru':
            self.space_dict['rnn_encoder'] = opt.rnn_size * 2
            self.txt_encoder_num += 1
            self.encoder.add_module('rnn_encoder', BiGruTxtEncoder(opt))
        elif self.rnn_encoding == 'nogru':
            pass

        # bert
        if self.bert_encoding == 'noBert':
            pass
        else:
            self.space_dict['bert_encoder'] = opt.bert_size
            self.txt_encoder_num += 1
            self.encoder.add_module('bert_encoder', BertTxtEncoder(opt))

        # w2v, bow
        if 'no' not in self.bow_encoding:
            self.space_dict['bow_encoder'] = opt.t2v_bow.ndims
            self.txt_encoder_num += 1
            self.encoder.add_module('bow_encoder', BoWTxtEncoder(opt))
        if 'no' not in self.w2v_encoding:
            self.space_dict['w2v_encoder'] = opt.t2v_w2v.ndims
            self.txt_encoder_num += 1
            self.encoder.add_module('w2v_encoder', W2VTxtEncoder(opt))

        # CLIP
        if 'no' not in self.CLIP_encoding:
            self.space_dict['CLIP_encoder'] = opt.clip_opt['size']
            self.txt_encoder_num += 1
            self.encoder.add_module('CLIP_encoder', CLIPEncoder(opt))

        # NetVLAD
        if 'no' not in self.NetVLAD_encoding:
            self.space_dict['NetVLAD_encoder'] = opt.t2v_w2v.ndims
            self.txt_encoder_num += 1
            self.encoder.add_module('NetVLAD_encoder', NetVLADTxtEncoder(opt))

        # encoder_names
        self.encoder_name_list = []
        for name, parm in self.encoder.named_modules():
            if '.' in name or name == '':  # if it is children name, continue
                continue
            self.encoder_name_list.append(name)

    def init_transform(self, opt, suffix=''):
        common_space_dim = opt.txt_fc_layers[1]
        self.attention_layer = Attention_1(common_space_dim)
        if 'transform_layer' not in dict(self.named_modules()):
            self.transform_layer = nn.Module()

        dropout = opt.dropout
        batch_norm = opt.batch_norm
        activation = opt.activation

        if 'no' not in self.rnn_encoding:
            if 'bigru' in self.rnn_encoding:
                rnn_transform = TransformNet((self.space_dict['rnn_encoder'], common_space_dim), None, dropout,
                                             batch_norm, activation)
                self.transform_layer.add_module('rnn_encoder' + '_transform' + suffix, rnn_transform)
            else:
                rnn_transform = TransformNet((self.space_dict['rnn_encoder'], common_space_dim), None, dropout,
                                             batch_norm, activation)
                self.transform_layer.add_module('rnn_encoder' + '_transform' + suffix, rnn_transform)

        if self.bert_encoding == 'noBert':
            pass
        else:
            bert_transform = TransformNet((opt.bert_size, common_space_dim), None, opt.bert_transform_dropout,
                                          opt.bert_transform_batch_norm, opt.bert_transform_activation)
            self.transform_layer.add_module('bert_encoder' + '_transform' + suffix, bert_transform)

        if 'no' not in self.w2v_encoding:
            w2v_transform = TransformNet((opt.t2v_w2v.ndims, common_space_dim), None, dropout,
                                         batch_norm, activation)
            self.transform_layer.add_module('w2v_encoder' + '_transform' + suffix, w2v_transform)

        if 'no' not in self.bow_encoding:
            bow_transform = TransformNet((opt.t2v_bow.ndims, common_space_dim), None, dropout,
                                         batch_norm, activation)
            self.transform_layer.add_module('bow_encoder' + '_transform' + suffix, bow_transform)

        if 'no' not in self.CLIP_encoding:
            if "CLIP_encoder" in self.opt.txt_no_transform:
                CLIP_transform = TransformNet((
                    opt.clip_opt['size'], common_space_dim), None,
                    opt.clip_opt['transform_dropout'], opt.clip_opt['transform_batch_norm'],
                    False, False)
            else:

                CLIP_transform = TransformNet((
                    opt.clip_opt['size'], common_space_dim), None,
                    opt.clip_opt['transform_dropout'], opt.clip_opt['transform_batch_norm'],
                    opt.clip_opt['transform_activation'])
            self.transform_layer.add_module('CLIP_encoder' + '_transform' + suffix, CLIP_transform)

        if 'no' not in self.NetVLAD_encoding:
            NetVLAD_transform = TransformNet((opt.t2v_w2v.ndims * opt.NetVLAD_opt['num_clusters'], common_space_dim),
                                             None, dropout,
                                             batch_norm, activation)
            self.transform_layer.add_module('NetVLAD_encoder' + '_transform' + suffix, NetVLAD_transform)

        if self.opt.txt_attention != 'concat':
            self.attention_layer = get_attention_layer(
                self.opt.txt_attention, common_space_dim, self.txt_encoder_num, self.opt)

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.init_txt_encoder(opt)

    def forward(self, caption_feat_dict, task3=False):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        out_feature = []
        encoder_module_dict = dict(self.encoder.named_modules())
        for name in self.encoder_name_list:
            txt_features = encoder_module_dict[name](caption_feat_dict, task3=task3)['text_features']
            txt_features = to_device_and_float16(txt_features)
            out_feature.append(txt_features)

        out = torch.cat(out_feature, dim=1)
        return out


class MultiScaleTxtNet(nn.Module):
    def _init_encoder(self, opt):
        self.encoder = MultiScaleTxtEncoder(opt)

    def _init_transformer(self, opt):
        self.transformer = TransformNet(
            opt.txt_fc_layers, opt, opt.dropout, opt.batch_norm,
            opt.activation)

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self._init_encoder(self.opt)

        self.opt.txt_fc_layers[0] = 0
        for name in self.encoder.space_dict:
            self.opt.txt_fc_layers[0] += self.encoder.space_dict[name]

        self._init_transformer(self.opt)

    def forward(self, caption_feat_dict, task3=False):
        features = self.encoder(caption_feat_dict, task3)
        features = self.transformer(features)
        return features


'''
class W2VV (CrossModalNetwork):
    def __init_vis_net(self, opt):
        self.vis_net = IdentityNet(opt)

    def __init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet(opt)

class W2VVPP (CrossModalNetwork):
    """
    w2v++ 最主要的net, self.vis_net为视频特征转换网络，
    self.txt_net为多网络拼接的 文本查询转换网络
    """
    def _init_vis_net(self, opt):
        self.vis_net = VisTransformNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet(opt)
'''


# ****************************萌萌哒分界线****************************************
class W2VVPP(nn.Module):
    """
    w2v++ 加入预测 Bert 作为 txt encoder

        w2v++ 最主要的net, self.vis_net为视频特征转换网络，可以使用 ircsn finetune 作为 vis 输入
        self.txt_net为多网络拼接的 文本查询转换网络
        """

    def _init_vis_net(self, opt):
        self.vis_net = VisTransformNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet(opt)
        if opt.txt_fc_same_with_vis_fc:
            if self.txt_net.transformer.fc1.weight.shape[1] == self.vis_net.fc1.weight.shape[1]:
                self.txt_net.transformer = self.vis_net
            else:
                raise Exception("txt_fc is not matching vis_fc ")

    def _init_neg_setting(self):
        opt = self.opt
        # ************************萌萌哒***************************
        # Negative
        self.kl = opt.kl
        if self.kl:
            self.klloss=KlLoss(cost_style=opt.cost_style, device=device,   direction=opt.direction)

        self.context_length=32
        self.task3_neg_retrival_weight=opt.task3_neg_retrival_weight
        self.criterion_task3 = Margin2Loss(neg_weight=opt.task3_neg_weight,bottommargin=opt.task3_bottommargin,uppermargin=opt.task3_uppermargin,
                                           bottommargin_t2t=opt.task3_bottommargin_t2t,uppermargin_t2t=opt.task3_uppermargin_t2t,
                                          measure=opt.measure,
                                          cost_style=opt.cost_style, device=device)
        self.criterion_with_score = MarginRankingLossWithScore(margin=opt.margin,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)
        # ************************萌萌哒***************************

    def __init__(self, opt):
        super().__init__()
        self.scaler = GradScaler()
        if opt is None:
            return
        self._init_vis_net(opt)
        self._init_txt_net(opt)

        self.opt = opt
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.criterion = MarginRankingLoss(margin=opt.margin,
                                           measure=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)

        self.params = list(self.parameters())  # 所有 params

        # 设置学习率
        params_special = []
        params_usual = []
        for name, parm in list(self.named_parameters()):
            if ('BertModel' in name) or ('csn_model' in name) or ('ClipModel' in name):
                params_special.append(parm)
            else:
                params_usual.append(parm)
        params = [{'params': params_usual},
                  {'params': params_special, 'lr': opt.lr / 20}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0
        # ************************萌萌哒***************************
        # Negative
        self._init_neg_setting()



    def compute_loss(self, vis_embs, txt_embs, vis_embs_multi_labels=0, txt_embs_multi_labels=0, labels_embs=0):
        """Compute the loss given pairs of image and caption embeddings
        """
        if len(vis_embs.shape) == len(txt_embs.shape) == 2:
            triplet_loss = self.criterion(txt_embs, vis_embs)
            multi_label_loss_vis = 0
            multi_label_loss_txt = 0
            multi_label_triplet_loss = 0
            loss = triplet_loss + multi_label_loss_vis + multi_label_loss_txt + multi_label_triplet_loss
            loss_items = {
                'triplet_loss': triplet_loss
            }
        elif len(vis_embs.shape) == len(txt_embs.shape) == 3:
            """
            [batch_size, head_num, emb_dim]
            """
            triplet_loss_multi_head = 0
            for each in range(vis_embs.size(1)):
                triplet_loss_multi_head += self.criterion(txt_embs[:, each, :], vis_embs[:, each, :])
            loss = triplet_loss_multi_head
            loss_items = {
                'triplet_loss': triplet_loss_multi_head
            }
        else:
            raise Exception("vis_embs dims are not equal to txt_embs dims")
        return loss, loss_items

    def cal_foward(self, train_data):
        (vis_input, caption_feat_dict, labels_input,
         vis_frame_feat_dict_input,
         vis_origin_frame_tuple) = (
            train_data['vis_feats'], train_data['captions'],
            train_data['captions_task2'], train_data['vis_frame_feat_dict'],
            train_data['vis_origin_frame_tuple']
        )
        if vis_frame_feat_dict_input == {}:
            vis_frame_feat_dict_input = None
        # compute the embeddings
        txt_embs = self.txt_net(caption_feat_dict)
        vis_embs = self.vis_net(vis_input, txt_emb=txt_embs,
                                vis_frame_feat_dict_input=vis_frame_feat_dict_input)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, loss_items = self.compute_loss(vis_embs, txt_embs, 0, 0, 0)
        # print("triplet_loss and multi_label_loss_vis", loss_items, end='\r')

        return loss, loss_items

    def cal_foward_neg(self, train_data, epoch=1):

        (vis_input, caption_feat_dict,
         vis_frame_feat_dict_input,
         vis_origin_frame_tuple,masktoken,maskpos,falsecaption,origin_maskpos,captions_neg_flag,origin_vis_feat) = (
            train_data['vis_feats'], train_data['captions'], train_data['vis_frame_feat_dict'],
            train_data['vis_origin_frame_tuple'], train_data['masktoken'],
            train_data['maskpos'],train_data['falsecaption'],train_data['origin_maskpos'],train_data['captions_task3_mask'],train_data['origin_vis_feat']
        )
        if vis_frame_feat_dict_input == {}:
            vis_frame_feat_dict_input = None
        # compute the visual embeddings and original text embs

        txt_embs = self.txt_net(caption_feat_dict)
        vis_embs = self.vis_net(vis_input, txt_emb=txt_embs,
                                vis_frame_feat_dict_input=vis_frame_feat_dict_input)

        origin_txt_feat = None
        #有反例的句子
        task3_index = np.where(captions_neg_flag > -1)[0]

        if epoch < self.opt.task3_end:
            # 错误query
            false_embs = self.txt_net(falsecaption)

            #保持原query保持不变
            if self.kl:
                origin_txt_feat = caption_feat_dict["origin_text_feats"]
            self.optimizer.zero_grad()

            loss, loss_items = self.compute_loss_neg(vis_embs, txt_embs, 0, 0, 0, false_embs,
                                                 captions_neg_flag,None
            ,None,None,None,origin_vis_feat,origin_txt_feat)
        else:
            self.optimizer.zero_grad()

            loss, loss_items = self.compute_loss_neg(vis_embs, txt_embs, 0, 0, 0, None,
                                                 captions_neg_flag,None,None)

        return loss, loss_items

    def compute_loss_neg(self, vis_embs, txt_embs, vis_embs_multi_labels, txt_embs_multi_labels, labels_embs,false_txt_embs,mask_task3 ,\
                     mask_prediction_scores=None, masked_lm_labels=None,mask_origin_prediction_scores=None,origin_masked_lm_labels=None,origin_vis_feat=None,origin_text_feats=None):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_items = self.compute_loss(txt_embs, vis_embs, 0, 0, 0)

        if false_txt_embs is not None:

            task3_index = np.where(mask_task3 > -1)[0]
            mask_task3 = mask_task3[task3_index]
            mask_task3 = torch.Tensor(mask_task3).to(device)
            triplet_loss_task3 = 0
            if len(vis_embs.shape) == len(txt_embs.shape) == 3:
                for each in range(vis_embs.size(1)):
                    triplet_loss_task3 += self.criterion_task3(
                        txt_embs[task3_index, each, :], vis_embs[task3_index, each, :],
                        false_txt_embs[:, each, :], mask_task3) / len(task3_index) * txt_embs.shape[0]
            else:
                triplet_loss_task3 = self.criterion_task3(txt_embs[task3_index,:],vis_embs[task3_index,:] , false_txt_embs[:, :],\
                                                          mask_task3)/len(task3_index)*txt_embs.shape[0]

            loss_items['triplet_loss_negation']=triplet_loss_task3
            loss += triplet_loss_task3 * self.task3_neg_retrival_weight

        # if origin_text_feats is not None:
        #
        #     origin_score = cosine_sim(to_device_and_float16(origin_text_feats[task3_index,:]),to_device_and_float16(origin_vis_feat))
        #     klloss=self.klloss(score[task3_index,:], origin_score)
        #     loss += klloss*self.opt.kl_weight
        #     loss_items['kl_loss'] = klloss


        return loss, loss_items

    def forward(self, train_data, epoch=None):
        """One training step given vis_feats and captions.
        """

        self.iters += 1

        if float16:
            # 前向过程(model + loss)开启 autocast
            with autocast():
                if self.opt.negative:
                    loss, loss_items = self.cal_foward_neg(train_data)
                else:
                    loss, loss_items = self.cal_foward(train_data)

            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                clip_grad_norm_(self.params, self.grad_clip)

            # scaler.step() unscale之前放大后的梯度，但是scale太多可能出现inf或NaN
            # 故其会判断是否出现了inf/NaN
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 如果检测到出现了inf或者NaN，就跳过这次梯度更新，同时动态调整scaler的大小
            self.scaler.step(self.optimizer)
            # 查看是否要更新scaler,这个要注意不能丢
            self.scaler.update()
        else:
            if self.opt.negative:
                loss, loss_items = self.cal_foward_neg(train_data)
            else:
                loss, loss_items = self.cal_foward(train_data)
            # compute gradient and do SGD step
            loss.backward()
            if self.grad_clip > 0:
                clip_grad_norm_(self.params, self.grad_clip)
            self.optimizer.step()

        return loss_items

    def get_txt2vis_matrix(self, txt_embs, vis_embs, measure='cosine'):
        if len(vis_embs.shape) == len(txt_embs.shape) == 2:
            txt2vis_sim = self.compute_sim(txt_embs, vis_embs, measure, device)

        elif len(vis_embs.shape) == len(txt_embs.shape) == 3:
            for j, each in enumerate(range(vis_embs.size(1))):
                txt2vis_sim_temp = self.compute_sim(txt_embs[:, each, :], vis_embs[:, each, :], measure,
                                                    device).unsqueeze(0)
                txt2vis_sims = txt2vis_sim_temp if j == 0 else torch.cat(
                    (txt2vis_sims, txt2vis_sim_temp), dim=0)

            txt2vis_sim = torch.mean(txt2vis_sims, dim=0)

        return txt2vis_sim

    @util.timer
    def predict(self, txt_loader, vis_loader, measure, record_emb=False):
        if vis_loader.dataset.length > 5e4:
            return self.predict_batch(txt_loader, vis_loader, measure, record_emb)
        self.eval()

        txt_ids = []
        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict).cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net(caption_feat_dict)

                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break
                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, self.vis_ids

    @util.timer
    def predict_batch(self, txt_loader, vis_loader, measure, record_emb=False):
        """
        predict similarity each batch.
        :param txt_loader:
        :param vis_loader:
        :param measure:
        :param record_emb:
        :return:
        """
        print("predict_batch !")
        self.eval()

        txt_ids = []
        vis_ids = []
        pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))

        with torch.no_grad():
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net(caption_feat_dict)  # a dict

                for j, output_dict in enumerate(vis_loader):

                    (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict']
                    )
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict)

                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    if i == 0:
                        vis_ids.extend(batch_vis_ids)

                    pbar.add(len(batch_vis_ids) * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, vis_ids

    @util.timer
    def predict_rerank(self, txt_loader, vis_loader, measure,
                       t2i_matrix,
                       topK=3000, k1=20, reranking_weight=2):
        def get_subset_embedding(vis_loader, video_index_subset_list):
            # 获取所有 visual embedding
            vis_subset = torch.utils.data.dataset.Subset(vis_loader.dataset, video_index_subset_list)
            vis_subset_dataloader = torch.utils.data.DataLoader(
                vis_subset, batch_size=vis_loader.batch_size, shuffle=False)
            # pbar = Progbar(len(vis_subset))
            vis_emb_all = None  # (vis_subset, embedding_size)
            for j, vis_subset_output_tuple in enumerate(vis_subset_dataloader):
                (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                    vis_subset_output_tuple[0], vis_subset_output_tuple[1],
                    vis_subset_output_tuple[2], vis_subset_output_tuple[3]
                )
                # pbar.update(len(idxs))
                with torch.no_grad():
                    vis_embs = self.vis_net(vis_input)
                    if vis_emb_all is None:
                        vis_emb_all = vis_embs
                        continue

                    if type(vis_embs) == dict:
                        for each in vis_embs:
                            #     vis_embs[each] = vis_embs[each].cpu()
                            vis_emb_all[each] = torch.cat((vis_emb_all[each], vis_embs[each]), 0)
                    elif type(vis_embs) == list or type(vis_embs) == tuple:
                        for index_vis, vis_emb in enumerate(vis_embs):
                            vis_emb_all[index_vis] = torch.cat((vis_emb_all[index_vis], vis_embs[index_vis]), 0)
                    else:
                        vis_emb_all = torch.cat((vis_emb_all, vis_embs), 0)
            return vis_emb_all

        scores_txt_vis_final = torch.Tensor(t2i_matrix)

        video_index_subset_lists = []
        inds = np.argsort(t2i_matrix, axis=1)
        for index in range(inds.shape[0]):
            ind = inds[index][::-1]
            video_index_subset_lists.append(ind[0:topK])

        # 获取每个 visual_batch 的三种矩阵
        print()
        pbar = Progbar(len(txt_loader.dataset))
        for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
            with torch.no_grad():
                txt_embs = self.txt_net(caption_feat_dict)  # a dict

            for j in range(len(txt_idxs)):
                pbar.add(1)
                # if type(txt_embs) == dict:
                #     txt_emb = {}
                #     for each in txt_embs:
                #         txt_emb[each] = txt_embs[each][j, :].unsqueeze(0)
                # else:
                #     txt_emb = txt_embs[j, :].unsqueeze(0)
                vis_topK_embs = get_subset_embedding(vis_loader, video_index_subset_lists[i * j + j])
                scores_txt_vis_s = self.get_txt2vis_matrix(txt_embs, vis_topK_embs, measure).cpu()
                scores_txt_vis = scores_txt_vis_s[j, :].unsqueeze(0)
                scores_txt_txt = torch.Tensor([[1]])
                scores_vis_vis = self.get_txt2vis_matrix(vis_topK_embs, vis_topK_embs, measure).cpu()

                # rerank
                scores_txt_vis_rerank = scores_txt_vis + torch.Tensor(reranking_weight * ReRank.re_ranking(
                    scores_txt_vis.numpy(), scores_txt_txt.numpy(), scores_vis_vis.numpy(), k1=k1
                ))

                scores_txt_vis_final[i * j + j, list(video_index_subset_lists[i * j + j])] = scores_txt_vis_rerank

        scores_txt_vis_final = l2norm(scores_txt_vis_final)
        return scores_txt_vis_final.numpy()

    @util.timer
    def predict_rerank_tkb_simple(self, txt_loader, vis_loader, measure,
                                  t2i_matrix,
                                  topK=3000, k1=20, reranking_weight=2):
        def get_subset_embedding(vis_loader, video_index_subset_list):
            # 获取所有 visual embedding
            vis_subset = torch.utils.data.dataset.Subset(vis_loader.dataset, video_index_subset_list)
            vis_subset_dataloader = torch.utils.data.DataLoader(
                vis_subset, batch_size=vis_loader.batch_size, shuffle=False)
            # pbar = Progbar(len(vis_subset))
            vis_emb_all = None  # (vis_subset, embedding_size)
            for j, vis_subset_output_tuple in enumerate(vis_subset_dataloader):
                (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                    vis_subset_output_tuple[0], vis_subset_output_tuple[1],
                    vis_subset_output_tuple[2], vis_subset_output_tuple[3]
                )
                # pbar.update(len(idxs))
                with torch.no_grad():
                    vis_embs = self.vis_net(vis_input)
                    if vis_emb_all is None:
                        vis_emb_all = vis_embs
                        continue

                    if type(vis_embs) == dict:
                        for each in vis_embs:
                            #     vis_embs[each] = vis_embs[each].cpu()
                            vis_emb_all[each] = torch.cat((vis_emb_all[each], vis_embs[each]), 0)
                    elif type(vis_embs) == list or type(vis_embs) == tuple:
                        for index_vis, vis_emb in enumerate(vis_embs):
                            vis_emb_all[index_vis] = torch.cat((vis_emb_all[index_vis], vis_embs[index_vis]), 0)
                    else:
                        vis_emb_all = torch.cat((vis_emb_all, vis_embs), 0)
            return vis_emb_all

        scores_txt_vis_final = torch.Tensor(t2i_matrix)

        video_index_subset_lists = []
        inds = np.argsort(t2i_matrix, axis=1)
        for index in range(inds.shape[0]):
            ind = inds[index][::-1]
            video_index_subset_lists.append(ind[0:topK])

        # 获取每个 visual_batch 的三种矩阵
        print()
        pbar = Progbar(len(txt_loader.dataset))
        for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
            with torch.no_grad():
                txt_embs = self.txt_net(caption_feat_dict)  # a dict

            for j in range(len(txt_idxs)):
                pbar.add(1)
                # if type(txt_embs) == dict:
                #     txt_emb = {}
                #     for each in txt_embs:
                #         txt_emb[each] = txt_embs[each][j, :].unsqueeze(0)
                # else:
                #     txt_emb = txt_embs[j, :].unsqueeze(0)
                vis_topK_embs = get_subset_embedding(vis_loader, video_index_subset_lists[i * j + j])
                scores_txt_vis_s = self.get_txt2vis_matrix(txt_embs, vis_topK_embs, measure).cpu()
                scores_txt_vis = scores_txt_vis_s[j, :].unsqueeze(0)
                scores_txt_txt = torch.Tensor([[1]])
                scores_vis_vis = self.get_txt2vis_matrix(vis_topK_embs, vis_topK_embs, measure).cpu()

                # rerank
                scores_txt_vis_rerank = scores_txt_vis + torch.Tensor(reranking_weight * ReRank.re_ranking_tkb_simple(
                    scores_txt_vis.numpy(), scores_txt_txt.numpy(), scores_vis_vis.numpy(), k1=k1, topK=topK,
                ))

                scores_txt_vis_final[i * j + j, list(video_index_subset_lists[i * j + j])] = scores_txt_vis_rerank

        scores_txt_vis_final = l2norm(scores_txt_vis_final)

        return scores_txt_vis_final.numpy()

    @util.timer
    def predict_rerank_tkb_visual_clip_sim(self, txt_loader, vis_loader, measure,
                                           t2i_matrix,
                                           topK=3000, k1=20, reranking_weight=2):
        # 开发中，使用 clip 作为 video 相似度矩阵
        def get_subset_embedding(vis_loader, video_index_subset_list):
            # 获取所有 visual embedding
            vis_subset = torch.utils.data.dataset.Subset(vis_loader.dataset, video_index_subset_list)
            vis_subset_dataloader = torch.utils.data.DataLoader(
                vis_subset, batch_size=vis_loader.batch_size, shuffle=False)
            # pbar = Progbar(len(vis_subset))
            vis_emb_all = None  # (vis_subset, embedding_size)
            for j, vis_subset_output_tuple in enumerate(vis_subset_dataloader):
                (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                    vis_subset_output_tuple[0], vis_subset_output_tuple[1],
                    vis_subset_output_tuple[2], vis_subset_output_tuple[3]
                )
                # pbar.update(len(idxs))
                with torch.no_grad():
                    vis_embs = self.vis_net(vis_input)
                    if vis_emb_all is None:
                        vis_emb_all = vis_embs
                        continue

                    if type(vis_embs) == dict:
                        for each in vis_embs:
                            #     vis_embs[each] = vis_embs[each].cpu()
                            vis_emb_all[each] = torch.cat((vis_emb_all[each], vis_embs[each]), 0)
                    elif type(vis_embs) == list or type(vis_embs) == tuple:
                        for index_vis, vis_emb in enumerate(vis_embs):
                            vis_emb_all[index_vis] = torch.cat((vis_emb_all[index_vis], vis_embs[index_vis]), 0)
                    else:
                        vis_emb_all = torch.cat((vis_emb_all, vis_embs), 0)
            return vis_emb_all

        scores_txt_vis_final = torch.Tensor(t2i_matrix)

        video_index_subset_lists = []
        inds = np.argsort(t2i_matrix, axis=1)
        for index in range(inds.shape[0]):
            ind = inds[index][::-1]
            video_index_subset_lists.append(ind[0:topK])

        # 获取每个 visual_batch 的三种矩阵
        print()
        pbar = Progbar(len(txt_loader.dataset))
        for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
            with torch.no_grad():
                txt_embs = self.txt_net(caption_feat_dict)  # a dict

            for j in range(len(txt_idxs)):
                pbar.add(1)
                # if type(txt_embs) == dict:
                #     txt_emb = {}
                #     for each in txt_embs:
                #         txt_emb[each] = txt_embs[each][j, :].unsqueeze(0)
                # else:
                #     txt_emb = txt_embs[j, :].unsqueeze(0)
                vis_topK_embs = get_subset_embedding(vis_loader, video_index_subset_lists[i * j + j])
                scores_txt_vis_s = self.get_txt2vis_matrix(txt_embs, vis_topK_embs, measure).cpu()
                scores_txt_vis = scores_txt_vis_s[j, :].unsqueeze(0)
                scores_txt_txt = torch.Tensor([[1]])
                scores_vis_vis = self.get_txt2vis_matrix(vis_topK_embs, vis_topK_embs, measure).cpu()

                # rerank
                scores_txt_vis_rerank = scores_txt_vis + torch.Tensor(reranking_weight * ReRank.re_ranking_tkb_simple(
                    scores_txt_vis.numpy(), scores_txt_txt.numpy(), scores_vis_vis.numpy(), k1=k1, topK=topK,
                ))

                scores_txt_vis_final[i * j + j, list(video_index_subset_lists[i * j + j])] = scores_txt_vis_rerank

        scores_txt_vis_final = l2norm(scores_txt_vis_final)
        return scores_txt_vis_final.numpy()

    @util.timer
    def predict_concept_rerank(self, txt_loader, vis_loader, measure,
                               video_index_list, topK=1000, Concept_weight=2, idf_log_base=np.e):
        self.eval()

        vis_subset = torch.utils.data.dataset.Subset(vis_loader.dataset, video_index_list)
        vis_subset_dataloader = torch.utils.data.DataLoader(
            vis_subset, batch_size=vis_loader.batch_size, shuffle=False)

        txt_ids = []
        vis_ids = []
        scores_txt_vis_final = None

        pbar = Progbar(len(txt_loader.dataset))

        # 按照 batch_txt 计算
        for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
            with torch.no_grad():
                txt_embs = self.txt_net(caption_feat_dict)  # a dict

            scores_txt_vis = None

            for j, vis_subset_output_tuple in enumerate(vis_subset_dataloader):

                (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                    vis_subset_output_tuple[0], vis_subset_output_tuple[1],
                    vis_subset_output_tuple[2], vis_subset_output_tuple[3]
                )

                with torch.no_grad():
                    vis_embs = self.vis_net(vis_input)

                score_txt_vis_batch = self.get_txt2vis_matrix(txt_embs, vis_embs, measure).cpu()
                scores_txt_vis = torch.cat((scores_txt_vis, score_txt_vis_batch),
                                           1) if scores_txt_vis is not None else score_txt_vis_batch

                if i == 0:
                    vis_ids.extend(batch_vis_ids)

            # rerank
            video_concept_pkl_path = '~/hf_code/VisualSearch/v3c1/tv19_20_21_a-photo-of-concept_txt2video_sim_matrix.pkl'
            Concept_re_ranking_class = ReRank.Concept_re_ranking(
                video_concept_pkl_path, video_index_list, scores_txt_vis.numpy(), caption_feat_dict['caption'],
                topK=topK, idf_log_base=idf_log_base,
            )
            scores_txt_vis_rerank = scores_txt_vis + Concept_weight * torch.Tensor(
                Concept_re_ranking_class.get_query_concept_sim_matrix())
            # scores_txt_vis_rerank = torch.Tensor(Concept_re_ranking_class.get_query_concept_sim_matrix())
            scores_txt_vis_final = torch.cat((scores_txt_vis_final, scores_txt_vis_rerank),
                                             0) if scores_txt_vis_final is not None else scores_txt_vis_rerank

            txt_ids.extend(batch_txt_ids)
            pbar.add(len(batch_txt_ids))

        scores_txt_vis_final = l2norm(scores_txt_vis_final)
        return scores_txt_vis_final.numpy(), txt_ids, vis_ids

    @util.timer
    def predict_adhoc(self, txt_loader, vis_loader, measure, record_emb=False):
        if vis_loader.dataset.length > 5e4:
            return self.predict_batch(txt_loader, vis_loader, measure, record_emb)
        self.eval()

        txt_ids = []
        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict).cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            labels = []
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids, video_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net(caption_feat_dict)
                labels.extend(video_ids)

                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break
                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, self.vis_ids, labels

    @util.timer
    def predictneg_adhoc(self, txt_loader, vis_loader, measure, record_emb=False, neg_method="sub"):
        """
        使用 bool 相减的方法去做实验
        :param txt_loader:
        :param vis_loader:
        :param measure:
        :param record_emb:
        :param neg_method:  bool search.
        :return:
        """
        if vis_loader.dataset.length > 5e4:
            return self.predict_batch(txt_loader, vis_loader, measure, record_emb)
        self.eval()

        txt_ids = []
        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict).cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            negscores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            negmasks = []
            labels = []
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids, neginfo, negmask, video_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net({'caption': caption_feat_dict['poscaps']})
                labels.extend(video_ids)
                negidx = [[txt_idxs[k]] for k in range(len(negmask)) if negmask[k] > 0]
                negmasks.extend(negmask)

                if len(neginfo) > 0:
                    negtxt_embs = self.txt_net({'caption': neginfo})
                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break
                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    if len(neginfo) > 0:
                        negscore = self.get_txt2vis_matrix(negtxt_embs, vis_embs, measure=measure).float()
                        negscore = negscore.clamp(min=0)
                        negscores[negidx, idxs] = negscore.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

            tempnegmask = torch.Tensor(negmasks).unsqueeze(1).repeat(1, len(self.vis_ids))
            scores=(scores+1)/2
            negscores = (negscores + 1) / 2
            if neg_method == "sub":
                # scores = scores - negscores * tempnegmask
                scores = scores - negscores
            elif neg_method == "mul":
                scores = scores*(1 - negscores)

        return scores.detach().numpy(), txt_ids, self.vis_ids, labels

    @staticmethod
    def compute_sim(query_embs, retro_embs, measure='cosine', device=torch.device('cuda')):
        query_embs = query_embs.to(device)
        retro_embs = retro_embs.to(device)
        if measure == 'cosine':
            return cosine_sim(query_embs, retro_embs)
        elif measure == 'hist':
            return hist_sim(query_embs, retro_embs)
        elif measure == 'euclidean':
            raise Exception('Not implemented')
        else:
            raise Exception('%s is invalid' % measure)

    @property
    def learning_rate(self):
        """Return learning rate"""
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list

    def lr_step(self, val_value):
        """
        降低学习率
        :param val_value:
        :return:
        """
        self.lr_schedulers[0].step()
        self.lr_schedulers[1].step(val_value)

    def change_raw_global_emb_weight(self):
        # 更改 raw_global_emb_weight 比例
        try:
            if hasattr(self.txt_net, 'attention_layer'):
                if hasattr(self.txt_net.attention_layer, 'get_raw_global_emb_weight'):
                    # 指数级别衰减
                    # new_global_emb_weight = self.opt.txt_attention_global_decay_rate * \
                    #                         self.txt_net.attention_layer.get_raw_global_emb_weight()
                    # self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)

                    # 线性衰减
                    new_global_emb_weight = self.opt.txt_attention_global_decay_rate - \
                                            1 + self.txt_net.attention_layer.get_raw_global_emb_weight()
                    if new_global_emb_weight < 0:
                        new_global_emb_weight = 0
                    self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                else:
                    print("txt_net.attention_layer doesn't have get_raw_global_emb_weight meathod")
                print("new_global_emb_weight: ", new_global_emb_weight)
            else:
                print("txt_net doesn't have attention_layer")
        except Exception as e:
            print(e)

        try:
            if hasattr(self.vis_net, 'attention_layer'):
                if hasattr(self.vis_net.attention_layer, 'get_raw_global_emb_weight'):
                    # 指数级别衰减
                    # new_global_emb_weight = self.opt.vis_attention_global_decay_rate * \
                    #                         self.vis_net.attention_layer.get_raw_global_emb_weight()
                    # self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                    # 线性衰减
                    new_global_emb_weight = self.opt.vis_attention_global_decay_rate - \
                                            1 + self.vis_net.attention_layer.get_raw_global_emb_weight()
                    if new_global_emb_weight < 0:
                        new_global_emb_weight = 0
                    self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                print("new_global_emb_weight: ", new_global_emb_weight)
            else:
                print("vis_net doesn't have attention_layer")
        except Exception as e:
            print(e)


class MultiScaleTxtEncoderAttention(MultiScaleTxtEncoder):
    """
    对 multi scale txt 加入 attention, 使用 vis_emb 或者 meanpooling 作为 attention.

    输入： txt_input
    输出： 各个 txt 特征，再映射到相同维度，再经过 attention 后特征。
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.init_transform(opt)

        # expert embeddings
        self.expert_embedding = None
        self.txt_expert_l2Norm = False
        if opt.txt_expert_embedding['expert']:
            common_space_dim = opt.txt_fc_layers[1]
            txt_expert_num = len(self.space_dict)
            self.expert_embedding = nn.Embedding(txt_expert_num, common_space_dim)
        if opt.txt_expert_embedding['l2norm']:
            self.txt_expert_l2Norm = True

    def forward(self, caption_feat_dict, visual_emb=None, task3=False):
        """Handles variable size captions
            local_embs: (batch, num_feature, dim)
            global_embs: (batch, num_heads, dim)
                """
        # Embed word ids to vectors
        local_emb_list = []

        encoder_module_dict = dict(self.encoder.named_modules())
        transform_layer_module_dict = dict(self.transform_layer.named_modules())
        for name in self.encoder_name_list:
            txt_features = encoder_module_dict[name](caption_feat_dict, task3=task3)['text_features']
            if name in self.opt.txt_no_transform:
                txt_features = txt_features.repeat(1, self.opt.multi_head_attention['heads'])
            txt_features = to_device_and_float16(txt_features)
            txt_subspace_features = transform_layer_module_dict[name + '_transform'](txt_features)

            # torch.isnan(txt_subspace_features).item()
            local_emb_list.append(txt_subspace_features)

        local_embs = torch.stack(local_emb_list, dim=1)

        # add expert_embedding
        if self.expert_embedding is not None:
            expert_index_tensor = torch.arange(
                0, local_embs.shape[1]).repeat(local_embs.shape[0], 1).to(device)
            expert_embs = self.expert_embedding(expert_index_tensor)
            local_embs = local_embs + expert_embs
            # print("We got expert_embedding!")
        # l2norm
        if self.txt_expert_l2Norm:
            local_embs = l2norm(local_embs, dim=2)

        if self.opt.txt_attention == 'attention':
            global_emb = self.attention_layer(local_embs, visual_emb)
        elif self.opt.txt_attention == 'con_attention':
            global_emb = self.con1_layer(local_embs)
            # global_emb = global_emb.squeeze(1) + torch.mean(local_embs, dim=1)
            global_emb = global_emb.squeeze(1)
        else:
            global_emb = self.attention_layer(local_embs)

        return global_emb

    def get_attention_weight(self, caption_feat_dict, visual_emb=None):
        self.forward(caption_feat_dict, visual_emb)
        return self.attention_layer.get_attention_weight()


class MultiScaleTxtEncoderList(MultiScaleTxtEncoderAttention):
    """
    对 multi scale txt 使用 fully connect 进行叠加
    输入：caption text.
    输出：一个各个txt特征映射到公共空间后特征的字典。
    """

    def __init__(self, opt):
        # todo: 照着 MultiScaleTxtEncoder 改，或者可以直接删除
        super().__init__(opt)
        self.attention_layer = None

    def forward(self, caption_feat_dict, visual_emb=None):
        """Handles variable size captions
                """
        # Embed word ids to vectors
        out_feature = None

        encoder_module_dict = dict(self.encoder.named_modules())
        transform_layer_module_dict = dict(self.transform_layer.named_modules())
        for name in self.encoder_name_list:
            txt_features = encoder_module_dict[name](caption_feat_dict)['text_features']
            txt_subspace_features = transform_layer_module_dict[name + '_transform'](txt_features)

            txt_subspace_features = txt_subspace_features.unsqueeze(1)
            out_feature = txt_subspace_features if out_feature is None \
                else torch.cat((out_feature, txt_subspace_features), dim=1)

        return out_feature


class VisTransformNetList(nn.Module):
    """
    输入：vis_input: 一个包含很多视频特征的字典。
    输出：out_features: 一个字典，key: txt_encoder name, value: 转换后视频特征。

    self.fuse_vis_feature_layer: 把多个视频特征融合网络
    """

    def __init__(self, opt, text_name_dict: dict):
        super().__init__()
        self.opt = opt
        if self.opt.vis_attention == 'concat':
            vis_transform_size = np.sum(list(opt.vis_fc_layers[0].values()))
            for each in text_name_dict.keys():
                self.add_module(each, TransformNet((vis_transform_size,
                                                    opt.vis_fc_layers[1]), opt))
        else:
            self.fuse_vis_feature_layer = VisMutiTransformNetAddAttnetion(opt, opt.vis_fc_layers[0])
            vis_transform_size = opt.vis_fc_layers[1]

        self.vis_net_space_dict = text_name_dict

    def forward(self, vis_input, txt_emb=None,
                vis_frame_feat_dict_input=None):
        out_features = {}
        if self.opt.vis_attention == 'concat':
            vis_feature = torch.cat(list(vis_input.values()), dim=1)

        else:
            for name in vis_input.keys():
                vis_input[name] = to_device_and_float16(vis_input[name])
                if len(torch.nonzero(vis_input[name])) == 0 and self.training:
                    # continue  ## 全0不能参加训练，否则会 nan
                    vis_input[name] = torch.randn_like(vis_input[name]).to(device)

            if self.opt.vis_feat_add_concat:
                if 'vis_feat_add_concat' not in vis_input:
                    vis_input['vis_feat_add_concat'] = torch.cat(list(vis_input.values()), dim=1)

            vis_feature = self.fuse_vis_feature_layer(vis_input)

        return vis_feature


class VisMutiTransformNet(nn.Module):
    """
    输入：vis features 字典。
    输出：转换成公共维度后 vis features 字典。
    """

    def __init__(self, opt, space_dict: dict):
        super().__init__()
        if opt == None:
            return
        self.opt = opt
        self.vis_net_space_dict = space_dict
        self.common_space_dim = opt.vis_fc_layers[1]
        for each in space_dict.keys():
            if each not in opt.vis_no_transform:
                self.add_module(each, TransformNet((space_dict[each], opt.vis_fc_layers[1]), opt))
            else:
                self.add_module(each, TransformNet((space_dict[each], opt.vis_fc_layers[1]), None, dropout=None,
                                                   batch_norm=True, activation=False, fc=False))

    def forward(self, vis_input, txt_emb=None, vis_frame_feat_dict_input=None):
        out_feature_dict = {}
        module_dict = dict(self.named_modules())

        if self.opt.vis_feat_add_concat:
            # for name in vis_input
            if 'vis_feat_add_concat' not in vis_input:
                vis_input['vis_feat_add_concat'] = torch.cat(list(vis_input.values()), dim=1)

        for name in self.vis_net_space_dict.keys():
            vis_input[name] = to_device_and_float16(vis_input[name])

            if len(torch.nonzero(vis_input[name])) == 0 and self.training:
                # continue  ## 全0不能参加训练，否则会 nan
                vis_input[name] = torch.randn_like(vis_input[name]).to(device)
            if name in self.opt.vis_no_transform:
                vis_input[name] = vis_input[name].repeat(1, self.opt.multi_head_attention['heads'])

            out_feature_dict[name] = module_dict[name](vis_input[name])

        return out_feature_dict


class VisMutiTransformNetAddAttnetion(nn.Module):
    """
    输入：vis features 字典。
    输出：(batchsize, common_space) 特征。
    """

    def __init__(self, opt, space_dict: dict):
        super().__init__()
        if opt == None:
            return
        self.opt = opt
        self.vis_net_space_dict = space_dict
        self.common_space_dim = opt.vis_fc_layers[1]
        self.VisMutiTransformNet = VisMutiTransformNet(opt, space_dict)

        self.attention_layer = get_attention_layer(
            self.opt.vis_attention, self.common_space_dim, len(space_dict), self.opt)

        # expert embeddings
        self.expert_embedding = None
        self.expert_l2Norm = False
        if opt.vis_expert_embedding['expert']:
            common_space_dim = self.common_space_dim
            vis_expert_num = len(self.vis_net_space_dict)
            self.expert_embedding = nn.Embedding(vis_expert_num, common_space_dim)
        if opt.vis_expert_embedding['l2norm']:
            self.expert_l2Norm = True

    def forward(self, vis_input, txt_emb=None, vis_frame_feat_dict_input=None):
        out_feature_dict = self.VisMutiTransformNet(vis_input)

        # 经过 attention 层转化为公共空间。
        vis_embs = torch.stack(list(out_feature_dict.values()), dim=1)

        # add expert_embedding
        local_embs = vis_embs
        if self.expert_embedding is not None:
            expert_index_tensor = torch.arange(
                0, local_embs.shape[1]).repeat(local_embs.shape[0], 1).to(device)
            expert_embs = self.expert_embedding(expert_index_tensor)
            local_embs = local_embs + expert_embs
        # l2norm
        if self.expert_l2Norm:
            local_embs = l2norm(local_embs, dim=2)

        vis_emb = self.attention_layer(local_embs)
        return vis_emb

    def get_attention_weight(self, vis_input, txt_emb=None):
        self.forward(vis_input, txt_emb)
        # return self.attention_layer.weights
        return self.attention_layer.get_attention_weight()


class W2VVPP_MutiVis(W2VVPP):
    """
    w2v++ 多个视频特征

        w2v++ 最主要的net, self.vis_net为视频特征转换网络，可以使用 ircsn finetune 作为 vis 输入
        self.txt_net为多网络拼接的 文本查询转换网络
        """

    def _init_txt_net(self, opt):
        if opt.txt_attention == 'concat':
            self.txt_net = MultiScaleTxtNet(opt)
        else:
            self.txt_net = MultiScaleTxtEncoderAttention(opt)

    def _init_vis_net(self, opt):
        if opt.vis_attention == 'concat':
            self.vis_net = VisTransformNet(opt)
        else:
            self.vis_net = VisMutiTransformNetAddAttnetion(opt, opt.vis_fc_layers[0])

    def __init__(self, opt):
        super().__init__(opt)

    def lr_step(self, val_value):
        super().lr_step(val_value)

    def change_raw_global_emb_weight(self):
        # 更改 raw_global_emb_weight 比例
        if hasattr(self.txt_net, 'attention_layer'):
            if hasattr(self.txt_net.attention_layer, 'get_raw_global_emb_weight'):
                # 指数级别衰减
                # new_global_emb_weight = self.opt.txt_attention_global_decay_rate * \
                #                         self.txt_net.attention_layer.get_raw_global_emb_weight()
                # self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)

                # 线性衰减
                new_global_emb_weight = self.opt.txt_attention_global_decay_rate - \
                                        1 + self.txt_net.attention_layer.get_raw_global_emb_weight()
                if new_global_emb_weight < 0:
                    new_global_emb_weight = 0
                self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
            else:
                print("txt_net.attention_layer doesn't have get_raw_global_emb_weight meathod")
        else:
            print("txt_net doesn't have attention_layer")

        if hasattr(self.vis_net, 'attention_layer'):
            if hasattr(self.vis_net.attention_layer, 'get_raw_global_emb_weight'):
                # 指数级别衰减
                # new_global_emb_weight = self.opt.vis_attention_global_decay_rate * \
                #                         self.vis_net.attention_layer.get_raw_global_emb_weight()
                # self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                # 线性衰减
                new_global_emb_weight = self.opt.vis_attention_global_decay_rate - \
                                        1 + self.vis_net.attention_layer.get_raw_global_emb_weight()
                if new_global_emb_weight < 0:
                    new_global_emb_weight = 0
                self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)


class W2VVPP_MultiHeadAttention(W2VVPP_MutiVis):
    """
    multi-head attention
    """

    def _init_txt_net(self, opt):
        if opt.txt_attention == 'concat':
            self.txt_net = MultiScaleTxtNet(opt)
        else:
            self.txt_net = MultiScaleTxtEncoderAttention(opt)
        if opt.txt_fc_same_with_vis_fc:
            for each in opt.txt_fc_same_with_vis_fc_dict:
                vis_module_dict = dict(self.vis_net.VisMutiTransformNet.named_modules())
                vis_name = opt.txt_fc_same_with_vis_fc_dict[each]
                if getattr(self.txt_net.transform_layer, each + '_transform').fc1.out_features == \
                        vis_module_dict[vis_name].fc1.out_features:

                    # setattr(self.txt_net.transform_layer, each + '_transform',
                    #         vis_module_dict[vis_name])
                    vis_module_dict[vis_name] = getattr(self.txt_net.transform_layer, each + '_transform')
                else:
                    raise Exception("txt_fc_same_with_vis_fc is not matching encoder_name_list")

        print()

    def _init_vis_net(self, opt):
        if opt.vis_attention == 'concat':
            self.vis_net = VisTransformNet(opt)
        else:
            self.vis_net = VisMutiTransformNetAddAttnetion(opt, opt.vis_fc_layers[0])

    def __init__(self, opt):
        super().__init__(None)
        if opt is None:
            return
        self.scaler = GradScaler()
        self.opt = opt
        self._init_vis_net(opt)
        self._init_txt_net(opt)

        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        if opt.loss == 'mrl':
            self.criterion = MarginRankingLoss(margin=opt.margin,
                                           measure=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)
        elif opt.loss == 'dsl':
            self.criterion = DualSoftmaxLoss()
        elif opt.loss == 'CELoss':
            self.criterion = CrossEntropyLoss()
        else:
            raise Exception("Not such loss.")

        self.criterion_with_score = MarginRankingLossWithScore(margin=opt.margin,
                                                               max_violation=opt.max_violation,
                                                               cost_style=opt.cost_style,
                                                               direction=opt.direction,
                                                               device=device)

        self.params = list(self.parameters())  # 所有 params

        # 设置学习率
        params_special = []
        params_usual = []
        for name, parm in list(self.named_parameters()):
            if ('BertModel' in name) or ('csn_model' in name) or ('ClipModel' in name):
                params_special.append(parm)
            else:
                params_usual.append(parm)
        params = [{'params': params_usual},
                  {'params': params_special, 'lr': opt.lr / 20}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, eps=1e-4)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0

    def compute_loss(self, vis_embs, txt_embs, vis_embs_multi_labels, txt_embs_multi_labels, labels_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        triplet_loss_multi_head = 0
        if self.opt.multi_space and len(vis_embs.shape) == len(txt_embs.shape) == 3:
            for each in range(vis_embs.size(1)):
                triplet_loss_multi_head += self.criterion(txt_embs[:, each, :], vis_embs[:, each, :])

        else:
            scores = self.get_txt2vis_matrix(txt_embs, vis_embs, self.opt.measure)
            triplet_loss_multi_head = self.criterion_with_score(scores)

        loss = triplet_loss_multi_head
        loss_items = {
            'triplet_loss': triplet_loss_multi_head
        }
        return loss, loss_items

    def get_txt2vis_matrix_each_head(self, txt_embs, vis_embs, measure):
        for j, each in enumerate(range(txt_embs.size(1))):
            txt2vis_sim_temp = self.compute_sim(txt_embs[:, each, :], vis_embs[:, each, :], measure,
                                                device).unsqueeze(0)
            txt2vis_sims = txt2vis_sim_temp if j == 0 else torch.cat(
                (txt2vis_sims, txt2vis_sim_temp), dim=0)

        txt2vis_sim = torch.sum(txt2vis_sims, dim=0)

        return txt2vis_sims
    @util.timer
    def predict_each_head(self, txt_loader, vis_loader, measure):


        self.eval()

        txt_ids = []
        vis_ids = []
        pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
        scores = torch.zeros((self.opt.multi_head_attention["heads"],len(txt_loader.dataset), len(vis_loader.dataset)))
        for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
            # if i > 1:
            #     txt_ids.extend(batch_txt_ids)
            #     continue

            with torch.no_grad():
                txt_embs = self.txt_net(caption_feat_dict)  # a dict

            for j, output_dict in enumerate(vis_loader):

                (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                    output_dict['vis_feat_dict'], output_dict['idxs'],
                    output_dict['vis_ids'], output_dict['vis_frame_feat_dict']
                )
                with torch.no_grad():
                    vis_embs = self.vis_net(vis_input)

                score = self.get_txt2vis_matrix_each_head(txt_embs, vis_embs, measure)

                scores[:, (i * txt_loader.batch_size):((i + 1) * txt_loader.batch_size), idxs] = score.cpu()

                if i == 0:
                    vis_ids.extend(batch_vis_ids)

                pbar.add(len(batch_vis_ids) * len(batch_txt_ids))

            txt_ids.extend(batch_txt_ids)

        return scores.numpy(), txt_ids, vis_ids


class VisMutiTransformNetPlusFrameFeat(nn.Module):
    """
    对 vid_frame_feats 进行 attention，计算结果再和 vid_feats 叠加输出进行 attenion。
    输入：vis features 字典, vis_frame_feat_dict_input 字典。
    输出：(batchsize, common_space) 特征。

    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        space_dict = opt.vis_fc_layers[0]
        if opt.frame_feat_input:
            for frame_name in opt.vid_frame_feats:
                space_dict[frame_name] = opt.vis_fc_layers[0][frame_name]
        self.vis_net_space_dict = space_dict
        for each in space_dict.keys():
            if each not in opt.vis_no_transform:
                self.add_module(each, TransformNet((space_dict[each], opt.vis_fc_layers[1]), opt))
            else:
                self.add_module(each, TransformNet((space_dict[each], opt.vis_fc_layers[1]), None, dropout=None,
                                                   batch_norm=True, activation=False, fc=False))

        common_space_dim = opt.vis_fc_layers[1]
        self.vis_attention_layer = get_attention_layer(
            self.opt.vis_attention, common_space_dim, len(space_dict), self.opt)

        # muti-frame features
        self.frame_attention = nn.ModuleDict()
        for each in opt.vid_frame_feats:
            common_space_dim = opt.vis_fc_layers[0][each]

            if self.opt.vis_frame_addFC:
                frame_attention_layer = nn.Sequential(
                    nn.Linear(common_space_dim, common_space_dim),
                    get_attention_layer(
                    self.opt.vis_frame_attention, common_space_dim, 1, self.opt))
            else:
                print("No Frame FC")
                frame_attention_layer = nn.Sequential(
                    get_attention_layer(
                    self.opt.vis_frame_attention, common_space_dim, 1, self.opt))

            self.frame_attention[each] = frame_attention_layer

    def forward(self, vis_input, vis_frame_feat_dict_input, txt_emb=None):
        """

        :param vis_input:
        :param vis_frame_feat_dict_input:  { [batch_size, max_length, embedding]}
        :param txt_emb:
        :return:
        """
        if self.opt.frame_feat_with_video_feat == False:
            vis_input = {}

        module_dict = dict(self.named_modules())
        # multi frame feats
        for feat_name in vis_frame_feat_dict_input:
            if feat_name == 'mask_tensor':
                continue
            # vis_frame_feat_dict_input[feat_name] [batch_size, frames, emb_size]
            batch_size, frames, emb_size = vis_frame_feat_dict_input[feat_name].shape
            vis_frame_feat_dict_input[feat_name] = to_device_and_float16(vis_frame_feat_dict_input[feat_name])

            for batch in range(batch_size):
                frame_input = vis_frame_feat_dict_input[feat_name][[batch]][0:int(vis_frame_feat_dict_input['mask_tensor'][0].sum().item())]
                video_emb = self.frame_attention[feat_name](frame_input).reshape(1, emb_size)
                if batch == 0:
                    vis_input[feat_name] = video_emb
                else:
                    vis_input[feat_name] = torch.cat((vis_input[feat_name], video_emb), dim=0)

            # vis_input[feat_name] = self.frame_attention[feat_name](vis_frame_feat_dict_input[feat_name])  # [batch_size, heads, emb_size]
            # vis_input[feat_name] = vis_input[feat_name].reshape(batch_size, emb_size)

        # multi visual feats
        out_feature_dict = {}
        for name in vis_input.keys():
            vis_input[name] = to_device_and_float16(vis_input[name])
            if name in self.opt.vis_no_transform:
                vis_input[name] = vis_input[name].repeat(1, self.opt.multi_head_attention['heads'])
            out_feature_dict[name] = module_dict[name](vis_input[name])

        # 经过 attention 层转化为公共空间。
        vis_embs = torch.stack(list(out_feature_dict.values()), dim=1)

        vis_emb = self.vis_attention_layer(vis_embs)
        return vis_emb

    def get_attention_weight(self, vis_input, vis_frame_feat_dict_input):
        self.forward(vis_input, vis_frame_feat_dict_input)
        return self.frame_attention[list(vis_frame_feat_dict_input.keys())[1]][0].get_attention_weight()


class W2VVPP_MutiVisFrameFeat(W2VVPP):
    """
    w2v++ 多个视频 frame 特征

        w2v++ 最主要的net, self.vis_net 为视频特征转换网络，可以使用 ircsn finetune 作为 vis 输入
        self.txt_net为多网络拼接的 文本查询转换网络
        """

    def _init_txt_net(self, opt):
        if opt.txt_attention == 'concat':
            self.txt_net = MultiScaleTxtNet(opt)
        else:
            self.txt_net = MultiScaleTxtEncoderAttention(opt)

    def _init_vis_net(self, opt):
        self.vis_net = VisMutiTransformNetPlusFrameFeat(opt)

    def __init__(self, opt):
        super().__init__(opt)

    def change_raw_global_emb_weight(self):
        # 更改 raw_global_emb_weight 比例
        try:
            if hasattr(self.txt_net, 'attention_layer'):
                if hasattr(self.txt_net.attention_layer, 'get_raw_global_emb_weight'):
                    # 指数级别衰减
                    # new_global_emb_weight = self.opt.txt_attention_global_decay_rate * \
                    #                         self.txt_net.attention_layer.get_raw_global_emb_weight()
                    # self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)

                    # 线性衰减
                    new_global_emb_weight = self.opt.txt_attention_global_decay_rate - \
                                            1 + self.txt_net.attention_layer.get_raw_global_emb_weight()
                    if new_global_emb_weight < 0:
                        new_global_emb_weight = 0
                    self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                else:
                    print("txt_net.attention_layer doesn't have get_raw_global_emb_weight meathod")
                print("new_global_emb_weight: ", new_global_emb_weight)
            else:
                print("txt_net doesn't have attention_layer")
        except Exception as e:
            print(e)

        try:
            if hasattr(self.vis_net, 'vis_attention_layer'):
                if hasattr(self.vis_net.vis_attention_layer, 'get_raw_global_emb_weight'):
                    # 指数级别衰减
                    # new_global_emb_weight = self.opt.vis_attention_global_decay_rate * \
                    #                         self.vis_net.attention_layer.get_raw_global_emb_weight()
                    # self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                    # 线性衰减
                    new_global_emb_weight = self.opt.vis_attention_global_decay_rate - \
                                            1 + self.vis_net.attention_layer.get_raw_global_emb_weight()
                    if new_global_emb_weight < 0:
                        new_global_emb_weight = 0
                    self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                print("new_global_emb_weight: ", new_global_emb_weight)
            else:
                print("vis_net doesn't have attention_layer")
        except Exception as e:
            print(e)


class End2EndClip(W2VVPP):
    """
    w2v端到端 clip model

        输入视频帧信息和原始文本，使用 clip 模型计算匹配。
        """

    def __init__(self, opt):
        super().__init__(None)
        if opt is None:
            return
        self.clip_model = CLIPEncoder(opt)
        self.clip_frozen = opt.clip_opt['frozen']

        self.opt = opt
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.criterion = MarginRankingLoss(margin=opt.margin,
                                           measure=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)

        self.params = list(self.parameters())  # 所有 params

        # 设置学习率
        params_special = []
        params_usual = []
        for name, parm in list(self.named_parameters()):
            if ('BertModel' in name) or ('csn_model' in name) or ('ClipModel' in name):
                params_special.append(parm)
            else:
                params_usual.append(parm)
        params = [{'params': params_usual},
                  {'params': params_special, 'lr': opt.lr / 100}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0

    @util.timer
    def predict(self, txt_loader, vis_loader, measure, record_emb=False):
        """
        :param txt_loader:
        :param vis_loader:
        :param measure:
        :param record_emb: record the video_all_embs and accelerate the prediction.
        :return:
        """
        self.eval()

        txt_ids = []

        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.clip_model(
                        caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                    )['visual_features'].cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.clip_model(caption_feat_dict)['text_features']

                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break

                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, self.vis_ids

    def cal_foward(self, train_data):
        (vis_input, caption_feat_dict, labels_input,
         vis_frame_feat_dict_input,
         vis_origin_frame_tuple) = (
            train_data['vis_feats'], train_data['captions'],
            train_data['captions_task2'], train_data['vis_frame_feat_dict'],
            train_data['vis_origin_frame_tuple']
        )
        if vis_frame_feat_dict_input == {}:
            vis_frame_feat_dict_input = None
        # compute the embeddings
        output = self.clip_model(caption_feat_dict,
                                   vis_origin_frame_tuple=vis_origin_frame_tuple,
                                )
        vis_embs, txt_embs = output['visual_features'], output['text_features']
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, loss_items = self.compute_loss(vis_embs, txt_embs, 0, 0, 0)
        # print("triplet_loss and multi_label_loss_vis", loss_items, end='\r')

        return loss, loss_items

    def forward(self, train_data, epoch=None):
        """One training step given vis_feats and captions.
        """
        if self.clip_frozen:
            self.iters += 1
            loss_items = {'triplet_loss': torch.Tensor(1)}
            self.optimizer.zero_grad()
            return loss_items
        else:
            return super().forward(train_data, epoch)

    def predictneg_adhoc(self, txt_loader, vis_loader, measure, record_emb=False, neg_method="sub"):
        self.eval()
        txt_ids = []

        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []
        labels=[]
        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    # if j>0:
                    #     break
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.clip_model(
                        caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                    )['visual_features'].cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            negscores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            negmasks = []
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids, neginfo, negmask, videoids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue
                labels.extend(videoids)
                negidx = [[txt_idxs[k]] for k in range(len(negmask)) if negmask[k] > 0]
                negmasks.extend(negmask)
                # text = to_device_and_float16(torch.cat(caption_feat_dict['clipcaption']))
                # txt_embs = self.clip_model.ClipModel.encode_text(text)
                txt_embs = self.clip_model({'caption': caption_feat_dict['poscaps']})['text_features']
                if len(neginfo) > 0:
                    negtxt_embs = self.clip_model({'caption': neginfo})['text_features']
                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break

                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if len(neginfo) > 0:
                        negscore = self.get_txt2vis_matrix(negtxt_embs, vis_embs, measure=measure).float()
                        negscore = negscore.clamp(min=0)

                    if i != len(txt_loader) - 1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    if len(neginfo) > 0:
                        negscores[negidx, idxs] = negscore.cpu()
                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)
            tempnegmask = torch.Tensor(negmasks).unsqueeze(1).repeat(1, len(self.vis_ids))
            scores=(scores+1)/2
            negscores = (negscores + 1) / 2
            if neg_method == "sub":
                scores = scores - negscores
            elif neg_method == "mul":
                scores = scores*(1 - negscores)
        return scores.detach().numpy(), txt_ids, self.vis_ids, labels


def get_model(name, device_, config):
    global device
    global float16
    device = device_
    float16 = config.float16

    NAME_TO_MODELS = {
        'W2VVPP': W2VVPP,
        'FrameLAFF': W2VVPP_MutiVisFrameFeat,
        'w2vpp_mutivis_attention': W2VVPP_MutiVis,
        'LAFF': W2VVPP_MultiHeadAttention,
        'End2EndClip': End2EndClip,

    }
    assert name in NAME_TO_MODELS, '%s not supported.' % name

    model_ = NAME_TO_MODELS[name](config)
    model_ = model_.float().to(device_)
    return model_


if __name__ == '__main__':
    global device

