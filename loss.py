# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def l2norm(X, eps=1e-13, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps + 1e-14
    X = torch.div(X, norm)
    return X


def l1norm(X, eps=1e-13, dim=1):
    """L2-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps + 1e-14
    X = torch.div(X, norm)
    return X


def normalization(X, dim=1):
    # 按行归一化
    _range = np.max(X) - np.min(X)
    return (X - np.min(X)) / _range


def cosine_sim(query, retrio):
    """Cosine similarity between all the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return query.mm(retrio.t())

def vector_cosine_sim(query, retrio):
    """Cosine similarity between  the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)
    return torch.sum(torch.mul(query, retrio),dim=1).unsqueeze(0)


def hist_sim(im, s, eps=1e-14):
    bs = im.size(0)
    im = im.unsqueeze(1).expand(-1, bs, -1)
    s = s.unsqueeze(0).expand(bs, -1, -1)
    intersection = torch.min(im, s).sum(-1)
    union = torch.max(im, s).sum(-1) + eps
    score = intersection / union
    return score


def jaccard_sim(query, retrieval_base, eps=1e-8):
    score = None
    base_num = retrieval_base.size(0)
    for each in query:
        each = each.unsqueeze(0).repeat(base_num, 1)
        intersection = torch.min(each, retrieval_base).sum(-1)
        union = torch.max(each, retrieval_base).sum(-1) + eps
        score_temp = (intersection / union).unsqueeze(0)
        if score is None:
            score = score_temp
        else:
            score = torch.cat((score, score_temp), dim=0)
    return score


class MarginRankingLoss(nn.Module):
    """
    Compute margin ranking loss
    arg input: (batchsize, subspace) and (batchsize, subspace)
    """
    def __init__(self, margin=0, measure='cosine', max_violation=False,
                 cost_style='sum', direction='bidir', device=torch.device('cpu')):
        """
        :param margin:
        :param measure: cosine 余弦相似度， hist_sim 扩展 jaccard 相似度
        :param max_violation:
        :param cost_style: 把所有误差相加 sum，还是取平均值 mean
        :param direction: compare every diagonal score to scores in its column and row
        """
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'hist':
            self.sim = hist_sim
        else:
            raise Exception('Not implemented.')

        self.max_violation = max_violation

    def forward(self, s, im):
        device = s.device
        # compute image-sentence score matrix
        scores = self.sim(im, s)  #
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)  # 扩展维度
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        I = I.to(device)

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)  # clamp 最大最小裁剪
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = torch.zeros(1).to(device)
        if cost_im is None:
            cost_im = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


class MarginRankingLossWithScore(nn.Module):
    """
    Compute margin ranking loss
    arg input: (batchsize, subspace) and (batchsize, subspace)
    """
    def __init__(self, margin=0, max_violation=False,
                 cost_style='sum', direction='bidir', device=torch.device('cpu')):
        """

        :param margin:
        :param measure: cosine 余弦相似度， hist_sim 扩展 jaccard 相似度
        :param max_violation:
        :param cost_style: 把所有误差相加 sum，还是取平均值 mean
        :param direction: compare every diagonal score to scores in its column and row
        """
        super().__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction

        self.max_violation = max_violation
        self.device = device

    def forward(self, score):
        device = self.device

        diagonal = score.diag().view(score.size(0), 1)
        d1 = diagonal.expand_as(score)  # 扩展维度
        d2 = diagonal.t().expand_as(score)

        # clear diagonals
        I = torch.eye(score.size(0)) > .5
        I = I.to(device)

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + score - d1).clamp(min=0)  # clamp 最大最小裁剪
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + score - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = torch.zeros(1).to(device)
        if cost_im is None:
            cost_im = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


class ImprovedBCELoss(nn.Module):
    def __init__(self, lambda_):
        super(ImprovedBCELoss, self).__init__()
        self.L = lambda_

    def forward(self, s, im):
        astype = torch.float
        im = im.type(astype)
        s = s.type(astype)
        weight_1 = self.L / torch.sum(im, dim=1, keepdim=True, dtype=astype) * im
        weight_2 = (1 - self.L) / torch.sum(1-im, dim=1, keepdim=True, dtype=astype) * (1-im)

        weight_1[weight_1 != weight_1] = 0  # NaN -> 0
        weight_2[weight_2 != weight_2] = 0

        res1 = torch.nn.functional.binary_cross_entropy_with_logits(s, im, weight=weight_1, reduction='sum')
        res2 = torch.nn.functional.binary_cross_entropy_with_logits(s, im, weight=weight_2, reduction='sum')

        return res1 + res2


class MarginLoss(nn.Module):
    """
    Compute margin  loss
    arg input: (batchsize, subspace) and (batchsize, subspace)
    """

    def __init__(self, neg_weight=1, margin=0, measure='cosine', cost_style='sum',
                 device=torch.device('cpu'), pos_weight=300):
        """

        :param margin:
        :param measure: cosine 余弦相似度， hist_sim 扩展 jaccard 相似度
        :param max_violation:
        :param cost_style: 把所有误差相加 sum，还是取平均值 mean
        :param direction: compare every diagonal score to scores in its column and row
        """
        super(MarginLoss, self).__init__()
        self.margin = 0
        self.cost_style = cost_style
        if measure == 'cosine':
            self.sim = vector_cosine_sim
        elif measure == 'hist':
            self.sim = hist_sim
        else:
            raise Exception('Not implemented.')

        self.device = device
        self.neg_weight = neg_weight

    def forward(self, txt_embs, vis_embs, false_txt_embs, weight):
        device = self.device
        # compute image-sentence score matrix
        scorest = self.sim(txt_embs, vis_embs)
        weight = weight * (self.neg_weight - 1) + 1

        scoresf = self.sim(false_txt_embs, vis_embs)
        cost = (self.margin + scoresf - scorest).clamp(min=0)

        cost = torch.mul(cost, weight).to(device)

        if self.cost_style == 'sum':

            return cost.sum()
        else:
            return cost.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, ):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, s, im, temp=1000):
        sim_matrix1 = cosine_sim(s, im)
        sim_matrix2 = sim_matrix1.T
        loss1 = self.cal_loss(sim_matrix1, temp)
        loss2 = self.cal_loss(sim_matrix2, temp)

        return (loss1 + loss2) / 2

    def cal_loss(self, sim_matrix):
        logpt = torch.diag(sim_matrix)
        logpt = torch.diag(logpt)
        loss = -logpt
        loss = loss.sum()
        return loss


class DualSoftmaxLoss(nn.Module):
    def __init__(self, ):
        super(DualSoftmaxLoss, self).__init__()

    def forward(self, s, im, temp=1000):
        sim_matrix1 = cosine_sim(s, im)
        sim_matrix2 = sim_matrix1.T
        loss1 = self.cal_loss(sim_matrix1, temp)
        loss2 = self.cal_loss(sim_matrix2, temp)

        return (loss1 + loss2) / 2

    def cal_loss(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(
            sim_matrix)  # With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        loss = loss.sum()
        return loss


class KlLoss(nn.Module):
    def __init__(self, cost_style='sum', direction='bidir', device=torch.device('cpu')):
        super().__init__()
        self.cost_style = cost_style
        self.direction = direction
        self.klloss= nn.KLDivLoss(reduction='none')
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax=nn.LogSoftmax(dim=1)
    def forward(self,score,originscore):

        losst2i = None

        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            originsimt2i = self.softmax(originscore)
            simt2i=self.logsoftmax(score)
            losst2i=self.klloss(simt2i,originsimt2i)


        if self.cost_style == 'sum':

            return losst2i.sum()
        else:
            return losst2i.mean()



class Margin2Loss(nn.Module):
    """
    Compute margin  loss
    arg input: (batchsize, subspace) and (batchsize, subspace)
    """

    def __init__(self, bottommargin,uppermargin, bottommargin_t2t,uppermargin_t2t, neg_weight=1, measure='cosine', cost_style='sum',
                 device=torch.device('cpu'), pos_weight=300):
        """

        :param margin:
        :param measure: cosine 余弦相似度， hist_sim 扩展 jaccard 相似度
        :param max_violation:
        :param cost_style: 把所有误差相加 sum，还是取平均值 mean
        :param direction: compare every diagonal score to scores in its column and row
        """
        super(Margin2Loss, self).__init__()
        self.uppermargin = uppermargin
        self.bottommargin=bottommargin
        self.uppermargin_t2t = uppermargin_t2t
        self.bottommargin_t2t=bottommargin_t2t
        self.cost_style = cost_style
        if measure == 'cosine':
            self.sim = vector_cosine_sim
        elif measure == 'hist':
            self.sim = hist_sim
        else:
            raise Exception('Not implemented.')

        self.device = device
        self.neg_weight = neg_weight

    def forward(self, txt_embs, vis_embs, false_txt_embs, weight):
        device = self.device
        # compute image-sentence score matrix
        scorest = self.sim(txt_embs, vis_embs)
        weight = weight * (self.neg_weight - 1) + 1
        scoresf = self.sim(false_txt_embs, vis_embs)
        scoresf2 = self.sim(false_txt_embs, txt_embs)
        cost=0
        if self.bottommargin  is not None:
            cost_b =  (self.bottommargin + scoresf - scorest).clamp(min=0)
            cost=cost+cost_b
        if self.uppermargin  is not None:
            cost_u = (-self.uppermargin - scoresf +scorest).clamp(min=0)
            cost=cost+cost_u
        if self.bottommargin_t2t  is not None:
            cost += (self.bottommargin_t2t + scoresf2 - scorest).clamp(min=0)
        if self.uppermargin_t2t  is not None:
            cost+=  (-self.uppermargin_t2t- scoresf2 +scorest).clamp(min=0)
        cost = torch.mul(cost, weight).to(device)

        if self.cost_style == 'sum':

            return cost.sum()
        else:
            return cost.mean()

