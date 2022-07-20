# -*- encoding: utf-8 -*-
import numpy as np
import random
import torch.nn as nn
import torch
from loss import l2norm, l1norm
import copy


class FcAttention(nn.Module):
    """
            Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
            Returns: - new_global: final embedding by FC, shape: (batch_size, embed_dim).
        """

    def __init__(self, embed_dim):
        super().__init__()
        self.fc_layer = nn.Linear(embed_dim, 1)

    def forward(self, local_embs: torch.Tensor, raw_global_emb=None):
        new_global = self.fc_layer(local_embs.permute((0, 2, 1)))
        new_global = new_global.squeeze(dim=2)
        return new_global


class JustAverage(nn.Module):
    """
            Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
            Returns: - new_global: final embedding by FC, shape: (batch_size, embed_dim).
        """

    def __init__(self):
        super().__init__()

    def forward(self, local_embs: torch.Tensor, raw_global_emb=None):
        new_global = torch.mean(local_embs, dim=1)
        return new_global


class Attention_1(nn.Module):
    """
        First version of self-attention.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: tagert embbeding, shape: (batch_size, embed_dim)
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, with_ave=True, mul=False):
        """

        :param embed_dim:
        :param with_ave: if 'attention_noAve' == True: 之后加上平均值
        :param mul: 没有 raw_global_emb 时是否进行 ave_global 与 local 相乘。
        """
        super().__init__()
        self.with_ave = with_ave
        self.mul = mul
        self.embed_dim = embed_dim
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.weights = 0  # attention 权重
        self.global_emb_weight_net = nn.Linear(1, 1, False)  # 存储 raw_global_emb 的权重
        self.change_raw_global_emb_weight(1)

    def get_raw_global_emb_weight(self):
        """
        得到 global_emb 的权重
        :return:
        """
        return self.global_emb_weight_net.weight.item()

    def change_raw_global_emb_weight(self, new_value: float):
        self.global_emb_weight_net.weight.data.fill_(new_value)

    def get_attention_weight(self):
        return torch.tensor(self.weights).clone().detach().cpu()

    def forward(self, local_embs: torch.Tensor, raw_global_emb=None):
        if raw_global_emb is None:
            common = local_embs  # (b, L, emb_size)
            raw_global_emb = torch.mean(local_embs, dim=1)

        if self.mul:
            # compute the normalized weights, shape: (batch_size, L)
            g_emb = raw_global_emb.unsqueeze(1).repeat(1, local_embs.size(1), 1)
            common = local_embs.mul(g_emb)  # (b, L, emb_size)

        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)
        self.weights = weights

        # compute final text, shape: (batch_size, 1024)
        new_global = weights.unsqueeze(2) * local_embs
        if self.with_ave:
            new_global_weights = 1
            raw_global_weight = self.get_raw_global_emb_weight()
            self.weights = new_global_weights * weights + raw_global_weight * 1.0 / weights.shape[1]  # weights + meanpooling
            # compute final text
            new_global = new_global_weights * new_global + raw_global_weight * torch.unsqueeze(raw_global_emb, 1)

        new_global = new_global.sum(dim=1)

        new_global = l2norm(new_global, eps=0)

        return new_global


class Attention_2(nn.Module):
    """
        My Multi heads of self-attention.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, opt=None):
        super().__init__()
        if opt is None:
            embed_dim_qkv = embed_dim
            dropout_rate = 0.1
            multi_heads = 1
        else:
            embed_dim_qkv = opt.multi_head_attention['embed_dim_qkv']
            dropout_rate = opt.multi_head_attention['dropout']
            multi_heads = opt.multi_head_attention['heads']
        self.embed_dim = embed_dim
        self.multi_heads = multi_heads

        self.embed_dim_qkv = embed_dim_qkv

        self.embedding_local_q = nn.Sequential()
        self.embedding_local_k = nn.Sequential()
        self.embedding_local_v = nn.Sequential()
        for i in range(multi_heads):
            self.embedding_local_q.add_module(str(i), nn.Sequential(nn.Linear(embed_dim, embed_dim_qkv),
                                                                    nn.Tanh(), nn.Dropout(dropout_rate)))
            self.embedding_local_k.add_module(str(i), nn.Sequential(nn.Linear(embed_dim, embed_dim_qkv),
                                                                    nn.Tanh(), nn.Dropout(dropout_rate)))
            self.embedding_local_v.add_module(str(i), nn.Sequential(nn.Linear(embed_dim, embed_dim_qkv),
                                                                    nn.Tanh(), nn.Dropout(dropout_rate)))
        self.embedding_common = nn.Sequential(nn.Linear(multi_heads * embed_dim_qkv, embed_dim))
        self.softmax = nn.Softmax(dim=1)

    def q_k_v_product_attention(self, q_emb, k_emb, v_emb):
        weights = torch.bmm(q_emb, k_emb.permute(0, 2, 1))  # (L, embed_dim_qkv) x (embed_dim_qkv, L) -> (L, L)
        weights = torch.div(weights, (self.embed_dim_qkv ** 0.5))
        weights = self.softmax(weights)
        new_v_emb = weights.bmm(v_emb)  # (L, L) x (L, embed_dim_qkv) -> (L, embed_dim_qkv)
        return new_v_emb

    def forward(self, local_embs, raw_global_emb=None):
        if raw_global_emb is None:
            raw_global_emb = torch.mean(local_embs, dim=1)

        new_v_embs = []
        for i in range(self.multi_heads):
            q_emb = self.embedding_local_q[i](local_embs)
            k_emb = self.embedding_local_k[i](local_embs)
            v_emb = self.embedding_local_v[i](local_embs)
            new_v_embs.append(self.q_k_v_product_attention(q_emb, k_emb, v_emb))
        new_v_emb = torch.cat(new_v_embs, dim=2)
        new_global = self.embedding_common(new_v_emb)  # (L, embed_dim_qkv)-> (L, embed_dim)

        new_global = new_global.sum(dim=1) + raw_global_emb
        new_global = l2norm(new_global, eps=1e-15)

        return new_global


class Attention_3(nn.Module):
    """
        My first one-head attention. Simple one head of self-attention.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, opt=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, embed_dim))
        self.softmax = nn.Softmax(dim=1)

    def q_k_v_product_attention(self, q_emb, k_emb, v_emb):
        weights = torch.bmm(q_emb, k_emb.permute(0, 2, 1))  # (L, embed_dim_qkv) x (embed_dim_qkv, L) -> (L, L)
        weights = torch.div(weights, (self.embed_dim ** 0.5))
        weights = self.softmax(weights)
        new_v_emb = weights.bmm(v_emb)  # (L, L) x (L, embed_dim_qkv) -> (L, embed_dim_qkv)
        return new_v_emb

    def forward(self, local_embs, raw_global_emb=None):
        if raw_global_emb is None:
            raw_global_emb = torch.mean(local_embs, dim=1)

        q_emb = local_embs
        k_emb = local_embs
        v_emb = local_embs
        new_v_emb = self.q_k_v_product_attention(q_emb, k_emb, v_emb)
        new_global = self.embedding_common(new_v_emb)  # (L, embed_dim)

        new_global = new_global.sum(dim=1) + raw_global_emb
        new_global = l2norm(new_global, eps=1e-15)

        return new_global


class Attention_multi_head_official(nn.Module):
    """
        Official multi-head attention.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - agg: how to aggragate the output of transformer. ('mean', 'max')
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, multi_heads=None, dropout_rate=None, agg='mean'):
        super().__init__()
        if multi_heads is None or dropout_rate is None:
            dropout_rate = 0.0
            multi_heads = 1

        self.agg = agg
        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim, multi_heads, dropout=dropout_rate, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
            vdim=None)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, local_embs, raw_global_emb=None):
        local_embs = local_embs.permute(1, 0,
                                        2)  # (batchsize, num_feature, embed_dim) -> (num_feature, batchsize, embed_dim)
        # self attention
        local_embs_attention, attention = self.attention_layer(local_embs, local_embs, local_embs,
                                                               key_padding_mask=None, )
        # add residual and norm layer
        new_global = self.layer_norm(local_embs + local_embs_attention)
        if self.agg == 'mean':
            new_global = torch.mean(new_global, 0)  # -> (batchsize, embed_dim)
        elif self.agg == 'max':
            new_global = torch.max(new_global, dim=0).values

        return new_global


class Attention_MMT(nn.Module):
    """
        Official multi-head attention.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, multi_heads=None, dropout_rate=None):
        super().__init__()
        if multi_heads is None or dropout_rate is None:
            dropout_rate = 0.0
            multi_heads = 1

        self.attention_layer = torch.nn.MultiheadAttention(
            embed_dim, multi_heads, dropout=dropout_rate, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
            vdim=None)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, local_embs, raw_global_emb=None):
        local_embs = local_embs.permute(1, 0,
                                        2)  # (batchsize, num_feature, embed_dim) -> (num_feature, batchsize, embed_dim)
        # Get agg
        local_emb_agg = torch.max(local_embs, dim=0, keepdim=True).values
        local_embs = torch.cat((local_emb_agg, local_embs), dim=0)

        # self attention
        local_embs_attention, attention = self.attention_layer(local_embs, local_embs, local_embs,
                                                               key_padding_mask=None, )
        # add residual and norm layer
        new_global = self.layer_norm(local_embs + local_embs_attention)
        new_global = new_global[0, :, :]  # -> (batchsize, embed_dim)

        return new_global


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)

        return context, attention


class Multi_head_MyApply_selfAttention(nn.Module):
    """
        my multi-head self-attention.
        使用官方 self-attention 思想，把多头的结果弄出来。
        参考：https://luozhouyang.github.io/transformer/
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - output_type: ['mean', 'max', 'first', 'last', 'cls_embedding',
               'max_embedding', 'mean_embedding', 'random', 'second', 'third', 'Attention_1']
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, num_heads, dim_per_head).
    """

    def __init__(self, embed_dim, multi_heads=None, dim_per_head=None,
                 dropout_rate=None, output_type='mean', encoder_num=0, l2norm_each_head=False,
                 opt=None):
        super().__init__()
        if dropout_rate is None:
            dropout_rate = 0.0
        assert dim_per_head == embed_dim // multi_heads

        self.dim_per_head = dim_per_head
        self.multi_heads = multi_heads

        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)

        self.layer_norm = nn.LayerNorm(dim_per_head)
        self.l2norm_each_head = l2norm_each_head

        # outpout type
        self.output_type = output_type
        if self.output_type == 'cls_embedding':
            self.cls_embedding = nn.Embedding(1, embed_dim)
        elif self.output_type == 'concat':
            self.concat_linear = nn.Linear(embed_dim * encoder_num, embed_dim)
        elif self.output_type == 'Attention_1':
            self.Attention_list = nn.ModuleList()
            for each in range(0, multi_heads):
                if opt is None:
                    attention_layer_each_head = Attention_1(dim_per_head)
                else:
                    attention_layer_each_head = Attention_1(
                        dim_per_head,
                        with_ave=opt.attention_param_each_head['with_ave'],
                        mul=opt.attention_param_each_head['mul'])
                self.Attention_list.append(attention_layer_each_head)


    def forward(self, local_embs, raw_global_emb=None, attn_mask=None):
        batch_size, L = local_embs.shape[0], local_embs.shape[1]
        if self.output_type == 'cls_embedding':
            cls_embedding = self.cls_embedding(torch.zeros((batch_size), dtype=torch.long, device=local_embs.device)).unsqueeze(1)
            cls_embedding = l2norm(cls_embedding, dim=2)
            local_embs = torch.cat((cls_embedding, local_embs), dim=1)
        elif self.output_type == 'concat':
            concat_embedding = self.concat_linear(local_embs.reshape(batch_size, -1)).unsqueeze(1)
            local_embs = torch.cat((concat_embedding, local_embs), dim=1)
        elif self.output_type == 'max_embedding':
            max_embedding = torch.max(local_embs, 1).values.unsqueeze(1)
            local_embs = torch.cat((max_embedding, local_embs), dim=1)
        elif self.output_type == 'mean_embedding':
            mean_embedding = torch.mean(local_embs, 1).unsqueeze(1)
            local_embs = torch.cat((mean_embedding, local_embs), dim=1)

        # self-attention
        key = value = query = local_embs  # (batch_size, L, embed_dim)
        batch_size = local_embs.size(0)
        num_heads = self.multi_heads
        dim_per_head = self.dim_per_head

        # split by heads, (batch_size, L, embed_dim) -> (batch_size*num_heads, L, dim_per_head)
        key = key.view(batch_size, -1, num_heads, dim_per_head
                 ).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head
                 ).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head
                 ).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        # key = key.view(batch_size * num_heads, -1, dim_per_head)
        # value = value.view(batch_size * num_heads, -1, dim_per_head)
        # query = query.view(batch_size * num_heads, -1, dim_per_head)

        if self.l2norm_each_head:
            query = l2norm(query, dim=2)
            value = l2norm(value, dim=2)
            key = l2norm(key, dim=2)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # add residual and norm layer
        output = self.layer_norm(context + query)
        output = output.view(batch_size, num_heads, -1, dim_per_head)
        if self.output_type == 'mean':
            output = torch.mean(output, 2)  # -> (batchsize, num_heads, embed_dim)
        elif self.output_type in ['first', 'cls_embedding',
                                  'concat', 'max_embedding', 'mean_embedding']:
            output = output[:, :, 0, :]  # -> (batchsize, num_heads, embed_dim)
        elif self.output_type == 'max':
            output = torch.max(output, 2).values
        elif self.output_type == 'last':
            output = output[:, :, -1, :]  # -> (batchsize, num_heads, embed_dim)
        elif self.output_type == 'random':
            if self.training:
                output = output[:, :, random.randint(0, L-1), :]  # -> (batchsize, num_heads, embed_dim)
            else:
                output = torch.mean(output, 2)
        elif self.output_type == 'second':
            if L < 2:
                output = output[:, :, 0, :]
            else:
                output = output[:, :, 1, :]  # -> (batchsize, num_heads, embed_dim)
        elif self.output_type == 'third':
            if L < 3:
                output = output[:, :, 0, :]
            else:
                output = output[:, :, 2, :]  # -> (batchsize, num_heads, embed_dim)
        elif self.output_type == 'Attention_1':
            output_Attention_1 = None
            for each in range(0, num_heads):
                output_each_head = output[:, each, :, :]
                output_each_head = self.Attention_list[each](output_each_head).unsqueeze(1)
                output_Attention_1 = output_each_head if output_Attention_1 is None else \
                    torch.cat((output_Attention_1, output_each_head), dim=1)
            output = output_Attention_1
        else:
            raise Exception("output_type error!")

        return output

    def get_raw_global_emb_weight(self):
        if hasattr(self, "Attention_list"):
            return self.Attention_list[0].get_raw_global_emb_weight()
        return 0

    def change_raw_global_emb_weight(self, new_value: float):
        if hasattr(self, "Attention_list"):
            print("Meanpooling weight: %.2f -> %.2f" % (self.Attention_list[0].get_raw_global_emb_weight(), new_value))
            for each in range(0, self.multi_heads):
                self.Attention_list[each].change_raw_global_emb_weight(new_value)
        return 0

    def get_attention_weight(self):
        if hasattr(self, "Attention_list"):
            weights = None
            for i in range(self.multi_heads):
                if weights is None:
                    weights = self.Attention_list[i].get_attention_weight()
                else:
                    weights += self.Attention_list[i].get_attention_weight()
            weights = weights / self.multi_heads
        else:
            raise Exception("SelfAttention weights is not applied")

        return weights


class Multi_head_MyApply_Attention(nn.Module):
    """
        my multi-head attention.
        使用官方 self-attention 思想，把多头的结果弄出来。再用 attention, average_attention,
        attention_noAve, attention_averageMul 等方法

        参考：https://luozhouyang.github.io/transformer/
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - split_head: whether split local_embs into each head.
        Returns: - new_global: final embedding by multi-head attention, shape: (batchsize, num_heads, dim_per_head).
    """

    def __init__(self, embed_dim, multi_heads=None, dim_per_head=None,
                 with_ave=True, mul=True, split_head=True, l2norm_each_head=False):
        super().__init__()
        if embed_dim is None:
            return

        self.dim_per_head = dim_per_head
        self.multi_heads = multi_heads
        self.split_head = split_head

        if self.split_head:
            assert dim_per_head == embed_dim // multi_heads
        else:
            dim_per_head = embed_dim
        self.attention_layer = nn.Sequential()
        for i in range(multi_heads):
            self.attention_layer.add_module(str(i), Attention_1(dim_per_head, with_ave=with_ave, mul=mul))

        self.layer_norm = nn.LayerNorm(dim_per_head)

        self.l2norm_each_head = l2norm_each_head

    def forward(self, local_embs, raw_global_emb=None, attn_mask=None):
        # self-attention
        key = value = query = local_embs  # (batch_size, L, embed_dim)
        batch_size = local_embs.size(0)
        num_heads = self.multi_heads
        dim_per_head = self.dim_per_head

        if self.split_head:
            # split by heads, (batch_size, L, embed_dim)-> (batch_size, L, num_heads, dim_per_head)
            local_embs = local_embs.view(batch_size, -1, num_heads, dim_per_head)
        else:
            # repeat by heads, (batch_size, L, embed_dim)-> (batch_size, heads, L, embed_dim)
            local_embs = local_embs.unsqueeze(2).repeat(1, 1, num_heads, 1)

        if self.l2norm_each_head:
            local_embs = l2norm(local_embs, dim=3)

        global_emb_list = []
        for i in range(self.multi_heads):
            global_emb_list.append(self.attention_layer[i](local_embs[:, :, i, :]))

        global_emb = torch.stack(global_emb_list, dim=1)  # -> (batchsize, num_heads, dim_per_head)

        return global_emb

    def get_raw_global_emb_weight(self):

        return self.attention_layer[0].global_emb_weight_net.weight.item()

    def change_raw_global_emb_weight(self, new_value: float):
        for i in range(self.multi_heads):
            self.attention_layer[i].global_emb_weight_net.weight.data.fill_(new_value)

    def get_attention_weight(self, head=0):

        weights = self.attention_layer[head].get_attention_weight().detach()

        # weights = None
        # for i in range(self.multi_heads):
        #     if weights is None:
        #         weights = self.attention_layer[i].get_attention_weight().detach()
        #     else:
        #         weights += self.attention_layer[i].get_attention_weight().detach()
        # weights = weights / self.multi_heads
        return weights


class Multi_head_My_AttentionMulGlobal(nn.Module):
    """
        my multi-head attention.
        使用官方 self-attention 思想，把多头的结果弄出来。再用 attention, average_attention,
        attention_noAve, attention_averageMul 等方法

        参考：https://luozhouyang.github.io/transformer/
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - split_head: whether split local_embs into each head.
        Returns: - new_global: final embedding by multi-head attention, shape: (batchsize, num_heads, dim_per_head).
    """

    def __init__(self, embed_dim, multi_heads=None, dim_per_head=None,
                 with_ave=True, mul=True, split_head=True, l2norm_each_head=False):
        super().__init__()
        if embed_dim is None:
            return

        self.dim_per_head = dim_per_head
        self.multi_heads = multi_heads
        self.split_head = split_head

        if self.split_head:
            assert dim_per_head == embed_dim // multi_heads
        else:
            dim_per_head = embed_dim
        self.attention_layer = nn.Sequential()
        for i in range(multi_heads):
            self.attention_layer.add_module(str(i), Attention_1(dim_per_head, with_ave=with_ave, mul=mul))

        self.layer_norm = nn.LayerNorm(dim_per_head)


    def forward(self, local_embs, raw_global_emb=None, attn_mask=None):
        # self-attention
        key = value = query = local_embs  # (batch_size, L, embed_dim)
        batch_size = local_embs.size(0)
        num_heads = self.multi_heads
        dim_per_head = self.dim_per_head

        if self.split_head:
            # split by heads, (batch_size, L, embed_dim)-> (batch_size, L, num_heads, dim_per_head)
            local_embs = local_embs.view(batch_size, -1, num_heads, dim_per_head)
            if raw_global_emb is not None:
                raw_global_emb = raw_global_emb.view(batch_size, num_heads, dim_per_head)
        else:
            # repeat by heads, (batch_size, L, embed_dim)-> (batch_size, heads, L, embed_dim)
            local_embs = local_embs.unsqueeze(2).repeat(1, 1, num_heads, 1)
            if raw_global_emb is not None:
                raw_global_emb = raw_global_emb.unsqueeze(1).repeat(1, num_heads, 1)

        global_emb_list = []
        for i in range(self.multi_heads):
            if raw_global_emb is not None:
                global_emb_list.append(self.attention_layer[i](local_embs[:, :, i, :], raw_global_emb=raw_global_emb[:, i, :]))
            else:
                global_emb_list.append(self.attention_layer[i](local_embs[:, :, i, :]))

        global_emb = torch.stack(global_emb_list, dim=1)  # -> (batchsize, num_heads, dim_per_head)

        return global_emb

    def get_raw_global_emb_weight(self):

        return self.attention_layer[0].global_emb_weight_net.weight.item()

    def change_raw_global_emb_weight(self, new_value: float):
        for i in range(self.multi_heads):
            self.attention_layer[i].global_emb_weight_net.weight.data.fill_(new_value)

    def get_attention_weight(self):
        weights = None
        for i in range(self.multi_heads):
            if weights is None:
                weights = self.attention_layer[i].get_attention_weight()
            else:
                weights += self.attention_layer[i].get_attention_weight()
        weights = weights / self.multi_heads
        return weights


class Multi_head_Attention_distinct_fc(Multi_head_MyApply_Attention):
    """
        my multi-head attention plus Attention. 每个特征一个全连接层。
        使用官方 self-attention 思想，把多头的结果弄出来。再用 attention, average_attention,
        attention_noAve, attention_averageMul 等方法

        参考：https://luozhouyang.github.io/transformer/
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - split_head: whether split local_embs into each head.
        Returns: - new_global: final embedding by multi-head attention, shape: (batchsize, num_heads, dim_per_head).
    """

    class Attention_distinct_fc(Attention_1):
        """
            each Attention has individual FC weights, and l2norm the local_embs.
            Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
                  - raw_global_emb: tagert embbeding, shape: (batch_size, embed_dim)
            Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
        """

        def __init__(self, embed_dim, with_ave=True, mul=False):
            """

            :param embed_dim:
            :param with_ave: if 'attention_noAve' == True: 之后加上平均值
            :param mul: 没有 raw_global_emb 时是否进行 ave_global 与 local 相乘。
            """
            super().__init__(embed_dim, with_ave, mul)
            self.with_ave = with_ave
            self.mul = mul
            self.embed_dim = embed_dim
            # self.embedding_common = nn.Sequential(
            #     nn.Linear(embed_dim, 128),
            #     nn.Tanh(),
            #     nn.Linear(128, 1),
            # )
            self.embedding_common = nn.ModuleList()
            for i in range(40):  # 最大40个特征
                self.embedding_common.append(nn.Linear(embed_dim, 1))

            self.softmax = nn.Softmax(dim=1)
            self.weights = 0  # attention 权重
            self.global_emb_weight_net = nn.Linear(1, 1, False)  # 存储 raw_global_emb 的权重
            self.change_raw_global_emb_weight(1)

        def forward(self, local_embs: torch.Tensor, raw_global_emb=None):
            if raw_global_emb is None:
                common = local_embs  # (b, L, emb_size)
                raw_global_emb = torch.mean(local_embs, dim=1)

            if self.mul:
                # compute the normalized weights, shape: (batch_size, L)
                g_emb = raw_global_emb.unsqueeze(1).repeat(1, local_embs.size(1), 1)
                common = local_embs.mul(g_emb)  # (b, L, emb_size)

            weights = None
            for each in range(common.shape[1]):
                if weights is None:
                    weights = self.embedding_common[each](common[:, each, :])
                else:
                    weights = torch.cat((weights, self.embedding_common[each](common[:, each, :])), dim=1)
            weights = self.softmax(weights)
            self.weights = weights

            # compute final text, shape: (batch_size, 1024)
            new_global = weights.unsqueeze(2) * local_embs
            if self.with_ave:
                new_global_weights = 1
                raw_global_weight = self.get_raw_global_emb_weight()
                self.weights = new_global_weights * weights + raw_global_weight * 1.0 / weights.shape[
                    1]  # weights + meanpooling
                # compute final text
                new_global = new_global_weights * new_global + raw_global_weight * torch.unsqueeze(raw_global_emb, 1)

            new_global = new_global.sum(dim=1)

            new_global = l2norm(new_global, eps=0)

            return new_global

    def __init__(self, embed_dim, multi_heads=None, dim_per_head=None,
                 with_ave=True, mul=True, split_head=True):
        super().__init__(None)
        if embed_dim is None:
            return

        self.dim_per_head = dim_per_head
        self.multi_heads = multi_heads
        self.split_head = split_head

        if self.split_head:
            assert dim_per_head == embed_dim // multi_heads
        else:
            dim_per_head = embed_dim
        self.attention_layer = nn.Sequential()
        for i in range(multi_heads):
            self.attention_layer.add_module(
                str(i),
                Multi_head_Attention_distinct_fc.Attention_distinct_fc(dim_per_head, with_ave=with_ave, mul=mul))


class Multi_head_Attention_layer_norm(Multi_head_MyApply_Attention):
    """
        my multi-head attention plus Attention_soft_l1.
        使用官方 self-attention 思想，把多头的结果弄出来。再用 attention, average_attention,
        attention_noAve, attention_averageMul 等方法

        参考：https://luozhouyang.github.io/transformer/
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - split_head: whether split local_embs into each head.
        Returns: - new_global: final embedding by multi-head attention, shape: (batchsize, num_heads, dim_per_head).
    """

    class Attention_layer_norm(Attention_1):
        """
            First use layer_norm.
            Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
                  - raw_global_emb: tagert embbeding, shape: (batch_size, embed_dim)
            Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
        """

        def __init__(self, embed_dim, with_ave=True, mul=False):
            """

            :param embed_dim:
            :param with_ave: if 'attention_noAve' == True: 之后加上平均值
            :param mul: 没有 raw_global_emb 时是否进行 ave_global 与 local 相乘。
            """
            super().__init__(embed_dim, with_ave, mul)
            self.layer_norm = nn.LayerNorm(embed_dim)

        def forward(self, local_embs: torch.Tensor, raw_global_emb=None):
            local_embs = self.layer_norm(local_embs)
            if raw_global_emb is None:
                common = local_embs  # (b, L, emb_size)
                raw_global_emb = torch.mean(local_embs, dim=1)

            if self.mul:
                # compute the normalized weights, shape: (batch_size, L)
                g_emb = raw_global_emb.unsqueeze(1).repeat(1, local_embs.size(1), 1)
                common = local_embs.mul(g_emb)  # (b, L, emb_size)

            weights = self.embedding_common(common).squeeze(2)
            weights = self.softmax(weights)
            self.weights = weights

            # compute final text, shape: (batch_size, 1024)
            new_global = weights.unsqueeze(2) * local_embs
            if self.with_ave:
                new_global_weights = (1 - self.get_raw_global_emb_weight())
                raw_global_weight = self.get_raw_global_emb_weight()
                self.weights = new_global_weights * weights + raw_global_weight * 1.0 / weights.shape[
                    1]  # weights + meanpooling
                # compute final text
                new_global = new_global_weights * new_global + raw_global_weight * torch.unsqueeze(raw_global_emb, 1)

            new_global = new_global.sum(dim=1)

            new_global = l2norm(new_global, eps=0)

            return new_global

    def __init__(self, embed_dim, multi_heads=None, dim_per_head=None,
                 with_ave=True, mul=True, split_head=True):
        super().__init__(None)
        if embed_dim is None:
            return

        self.dim_per_head = dim_per_head
        self.multi_heads = multi_heads
        self.split_head = split_head

        if self.split_head:
            assert dim_per_head == embed_dim // multi_heads
        else:
            dim_per_head = embed_dim
        self.attention_layer = nn.Sequential()
        for i in range(multi_heads):
            self.attention_layer.add_module(
                str(i), Multi_head_Attention_layer_norm.Attention_layer_norm(dim_per_head, with_ave=with_ave, mul=mul))

        self.layer_norm = nn.LayerNorm(dim_per_head)


class Multi_head_MyApply_FusionAttention(Multi_head_MyApply_Attention):
    """
        my multi-head attention plus Fusion Attention.
        每个 head 不同的 attention
        
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: target embbeding, shape: (batch_size, embed_dim)
              - split_head: whether split local_embs into each head.
        Returns: - new_global: final embedding by multi-head attention, shape: (batchsize, num_heads, dim_per_head).
    """

    def __init__(self, embed_dim, multi_heads=None, dim_per_head=None,
                 split_head=True):
        super().__init__(None)
        if embed_dim is None:
            return

        self.dim_per_head = dim_per_head
        self.multi_heads = multi_heads
        self.split_head = split_head

        if self.split_head:
            assert dim_per_head == embed_dim // multi_heads
        else:
            dim_per_head = embed_dim
        self.attention_layer = nn.Sequential()
        for i in range(multi_heads):
            if i % 4 == 0:
                self.attention_layer.add_module(str(i), Attention_1(dim_per_head, with_ave=True, mul=True))
            elif i % 4 == 1:
                self.attention_layer.add_module(str(i), Attention_1(dim_per_head, with_ave=True, mul=False))
            elif i % 4 == 2:
                self.attention_layer.add_module(str(i), Attention_1(dim_per_head, with_ave=False, mul=True))
            elif i % 4 == 3:
                self.attention_layer.add_module(str(i), Attention_1(dim_per_head, with_ave=False, mul=False))

        self.layer_norm = nn.LayerNorm(dim_per_head)


class NetVLAD(nn.Module):

    def __init__(self, feature_dim, num_clusters=32, alpha=100):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = feature_dim
        self.alpha = alpha
        print('num_clusters:', self.num_clusters)
        print('alpha:', self.alpha)

        init_sc = (1 / np.sqrt(feature_dim))
        self.fc1 = nn.Linear(self.dim, self.num_clusters, bias=False)
        #self.centeroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))
        self.centeroids = nn.Parameter(init_sc * torch.randn(self.num_clusters, self.dim))
        self.fc1.weight = nn.Parameter(init_sc * torch.randn(self.num_clusters, self.dim))

        #self._init_params()

    def _init_params(self):
        pass
        #self.fc1.weight = nn.Parameter(2.0 * self.alpha * self.centeroids)
        #self.fc1.bias = nn.Parameter(- self.alpha * self.centeroids.norm(dim=1))

    def forward(self, x):
        """Handles variable size video frames' feature
        params: x: list of _M*D features, 1 <= _M <= M
        """
        vlad = []
        for x_i in x:
            M, D = x_i.size()

            x_i = torch.nn.functional.normalize(x_i, p=2, dim=-1)  # across descriptor dim

            # soft-assignment
            soft_assign = self.fc1(x_i)  # M*K
            soft_assign = torch.nn.functional.softmax(soft_assign, dim=-1)  # M*K

            # M*K*D
            residual = x_i.expand(self.num_clusters, -1, -1).permute(1,0,2) - \
                    self.centeroids.expand(M, -1, -1)   # M*K*D
            residual *= soft_assign.unsqueeze(-1)

            vlad.append(residual.sum(dim=0))

        # N*K*D
        vlad = torch.stack(vlad, 0)

        vlad = torch.nn.functional.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(vlad.size(0), -1)  # flatten
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


if __name__ == '__main__':
    # official_MHA = Attention_multi_head_official(2048, 4)
    # official_MHA_param = sum(p.numel() for p in official_MHA.parameters())
    # print("official_MHA_param: \t%.4f M" % ((official_MHA_param / 1000000.0)))
    #
    # for i in [1, 2, 4, 8]:
    #     MyMulti_head = Multi_head_MyApply_Attention(2048, i, None, split_head=False)
    #     MyMulti_head_param = sum(p.numel() for p in MyMulti_head.parameters())
    #     print("MyMulti_head_param * %d: \t%.4f M" % (i, (MyMulti_head_param / 1000000.0)))

    from thop import profile

    for i in [2, 4, 8]:
        ave = JustAverage()
        official_MHA = torch.nn.TransformerEncoderLayer(
            2048, 8)
        MyMulti_head = Multi_head_MyApply_Attention(2048, 8, 256, split_head=True)

        features = torch.randn((1, i, 2048))
        flops1, params = profile(official_MHA, (features, ))
        flops2, params = profile(MyMulti_head, (features, ))
        flops3, params = profile(ave, (features, ))
        print("%.3f\t%.3f\t%.3f\n" % (flops1 / 1000 ** 2, flops2 / 1000 ** 2, flops3 / 1000 ** 2))
