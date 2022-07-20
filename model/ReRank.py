# -*- encoding: utf-8 -*-
import numpy as np
import pickle
import torch
import os
import re
import sys
sys.path.append('../')
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
from txt2vec import get_txt2vec
import util
import loss


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    API
        q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
        q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
        g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
        k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
    Returns:
        final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]

      CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking

    """

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    def k_reciprocal_neigh(initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]  # 第i个图片的前20个相似图片的索引号
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]  # 返回backward_k_neigh_index中等于i的图片的行索引号
        return forward_k_neigh_index[fi]  # 返回与第i张图片 互相为k_reciprocal_neigh的图片索引号

    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = 2. - 2 * original_dist  # np.power(original_dist, 2).astype(np.float32) 余弦距离转欧式距离
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))  # 归一化
    V = np.zeros_like(original_dist).astype(np.float32)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))  # 取前20，返回索引号

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)  # 取出互相是前20的
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):  # 遍历与第i张图片互相是前20的每张图片
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        # 增广k_reciprocal_neigh数据，形成k_reciprocal_expansion_index
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # 避免重复，并从小到大排序
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])  # 第i张图片与其前20+图片的权重
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(
            weight)  # V记录第i个对其前20+个近邻的权重，其中有0有非0，非0表示没权重的，就似乎非前20+的

    original_dist = original_dist[:query_num, ]  # original_dist裁剪到 只有query x query+g
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):  # 遍历所有图片
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)  # 第i张图片在initial_rank前k2的序号的权重平均值
            # 第i张图的initial_rank前k2的图片对应全部图的权重平均值
            # 若V_qe中(i,j)=0，则表明i的前k2个相似图都与j不相似
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def re_ranking_tkb_simple(q_g_dist, q_q_dist, g_g_dist, topK=3000, k1=20):
    """
    这种重排需要 video 数目较多，TopK 较大。
    :param q_g_dist: query-video similarity matrix
    :param q_q_dist:
    :param g_g_dist: 视频相似度矩阵
    :param k1: 每个视频找和它最相似的 k1 个
    :return:
    """

    video_indexs = np.arange(0, q_g_dist.shape[1])
    query_indexs = np.arange(0, q_g_dist.shape[0])

    # 找每个视频最近的k1个视频，视频出现数目组成字典
    video_sim_topK1 = {}
    inds = np.argsort(g_g_dist, axis=1)
    for index in range(inds.shape[0]):
        if index not in video_sim_topK1:
            video_sim_topK1[index] = 1
        else:
            video_sim_topK1[index] += 1
        ind = inds[index][::-1][0:k1]
        for each in ind:
            if each not in video_sim_topK1:
                video_sim_topK1[each] = 0
            video_sim_topK1[each] += 1


    # 对于每个 query 前 topK 个视频，进行重排
    # 重排依据：1. video_sim_topK1 视频出现次数；2. 原先次序
    q_g_dist_rerank = np.zeros(q_g_dist.shape)
    inds = np.argsort(q_g_dist, axis=1)
    for index in range(inds.shape[0]):
        ind_topK = inds[index][::-1][0:topK]  # 对于每个 query 前 topK 个视频

        # 在 topK 个视频中，找每个视频最近的k1个视频，视频出现数目组成字典
        # video_sim_topK1 = np.zeros(q_g_dist.shape[1])
        # g_g_dist_zero = np.zeros(g_g_dist.shape)
        # g_g_dist_zero[ind_topK, ind_topK] = g_g_dist[ind_topK, ind_topK]
        # g_g_dist_zero_inds = np.argsort(g_g_dist_zero, axis=1)
        # for index_k1 in ind_topK:
        #     ind_k1 = g_g_dist_zero_inds[index_k1][::-1][0:k1]
        #     for each in ind_k1:
        #         video_sim_topK1[each] += 1

        for each in ind_topK:
            try:
                q_g_dist_rerank[index, each] = np.log(video_sim_topK1[each] + 1)  # 相似度为 log(视频出现次数+1)
            except:
                print()

    q_g_dist_rerank = loss.l2norm(torch.Tensor(q_g_dist_rerank)).numpy()
    return q_g_dist_rerank

class Concept_re_ranking:
    def __init__(self,
                 video_concept_pkl_path, video_index_list, model_sim_matrix:np.ndarray, query_txts,
                 topK=2000, idf_log_base=np.e,
                 bow_nsw_path='~/hf_code/VisualSearch/tgif-msrvtt10k-vatex/TextData/vocab/bow_nsw_5.txt',
                 caption_path='~/hf_code/VisualSearch/tgif-msrvtt10k-vatex/TextData/tgif-msrvtt10k-vatex.caption.txt',
                 ):
        """

        :param video_concept_pkl_path: clip 计算的 video_concept 相似度矩阵。'~/VisualSearch/v3c1/tv19_20_21_a-photo-of-concept_txt2video_sim_matrix.pkl'
        :param video_index_list:
        :param model_sim_matrix: model 给出的相似度矩阵 (query_num, len(video_id_list))
        :param query_index:
        """
        # 路径修改成绝对路径
        video_concept_pkl_path = video_concept_pkl_path.replace('~', os.path.expanduser('~'))
        bow_nsw_path = bow_nsw_path.replace('~', os.path.expanduser('~'))
        caption_path = caption_path.replace('~', os.path.expanduser('~'))
        # 超参数
        self.idf_log_base = idf_log_base

        # 获得给定 video_list 的 concept space 相似度矩阵
        video_concept_array, concept_ids = self.get_concept(video_concept_pkl_path, video_index_list)
        self.concept_ids = concept_ids
        self.video_concept_array = self.video_concept_array_prepercess(video_concept_array,
                                                                       concept_ids, bow_nsw_path, caption_path)

        self.model_sim_matrix = model_sim_matrix
        self.query_topK_video_index_dict = self.get_query_topK_video_index_dict(model_sim_matrix, topK)  # index -> topK_index
        self.query_list = self.query_precess(query_txts)
        self.query_concept_matrix = self.get_query_concept_matrix()

    @util.timer
    def get_concept(self, pkl_path, video_index_list):
        """

        :param pkl_path:
        :param video_index_list:
        :return:  得到 video_id_list-concept 得分矩阵，以及 concept2index 字典
        """
        # pkl 文件是一个字典，里面是
        # {'txt2video_cos_sim_matrix': t2i_matrix, "txt_ids": np.array(txt_ids) , "vis_ids": np.array(vis_ids)}
        with open(pkl_path, 'rb') as f:
            video_concept_dict = pickle.load(f)
        vis_ids = video_concept_dict['vis_ids']
        concept_ids = video_concept_dict['txt_ids']
        # concept2index 字典
        # concept2index_dict = {}
        # for i, each in enumerate(concept_ids):
        #     concept2index_dict[each] = i

        sim_matrix = video_concept_dict['txt2video_cos_sim_matrix']

        # video_id_need_array = np.array(video_id_list)
        #
        # video_index_need = []
        # for ind, each in enumerate(video_id_need_array):
        #     video_index_need.append(np.where(vis_ids == each)[0][0])

        return sim_matrix[:, video_index_list].T, concept_ids

    @util.timer
    def get_query_topK_video_index_dict(self, model_sim_matrix, topK):
        query_topK_video_index_dict = {}
        if topK > model_sim_matrix.shape[1]:
            print("topK is larger than video_num")

        inds = np.argsort(model_sim_matrix, axis=1)  # 从小到大
        for index in range(inds.shape[0]):
            ind = inds[index][::-1][0:topK]  # 从大到小
            query_topK_video_index_dict[index] = ind

        return query_topK_video_index_dict

    @util.timer
    def video_concept_array_prepercess(
            self, video_concept_array: np.ndarray, concept_ids,
            bow_nsw_path='/data/liupengju/hf/tgif-msrvtt10k-vatex/TextData/vocab/bow_nsw_5.txt',
            caption_path='/data/liupengju/hf/tgif-msrvtt10k-vatex/TextData/tgif-msrvtt10k-vatex.caption.txt'
    ):
        """

        :param video_concept_array: (video_num, concept_num)
        :param concept_ids: ['dog', '', ...]
        :param bow_nsw_path:
        :param caption_path:
        :return:
        """
        # todo: video_concept_array 归一化

        # **************idf 加权**************
        caption_data = open(caption_path).read()
        txt_frequence_dict = {}
        with open(bow_nsw_path, 'r') as f_bow:
            for each in f_bow:
                text, frequence = each.strip().split(' ')[0], int(each.strip().split(' ')[1])
                txt_frequence_dict[text] = frequence
        concept2frequence_dict = {}
        for concept in concept_ids:
            if concept in txt_frequence_dict:
                concept2frequence_dict[concept] = txt_frequence_dict[concept]
            else:  # 计算 concept 在 caption_data 中出现的次数
                concept2frequence_dict[concept] = caption_data.count(concept)
                # if concept2frequence_dict[concept] == 0:
                #     print(concept)

        # 计算 idf：ln((1+出现总次数)/(1+concept出现次数))
        concept2idf_list = []
        sun_num = sum(list(concept2frequence_dict.values()))
        for concept in concept_ids:
            idf_raw = (1 + sun_num) / (concept2frequence_dict[concept] + 1)
            concept2idf_list.append(
                np.log(idf_raw) / np.log(self.idf_log_base)
            )

        # 计算 idf 加权后的 video_concept_array
        video_concept_array_idf = video_concept_array * \
                                  np.array(concept2idf_list).reshape(1, -1).repeat(video_concept_array.shape[0], axis=0)


        self.concept2frequence_dict = concept2frequence_dict
        return video_concept_array_idf

    def get_idf(
            self, query_text
    ):
        query_text_idf_dict = {}
        for concept in self.concept_ids:
            if concept in query_text:
                sun_num = sum(list(self.concept2frequence_dict.values()))
                idf_raw = (1 + sun_num) / (self.concept2frequence_dict[concept] + 1)
                query_text_idf_dict[concept] = np.log(idf_raw) / np.log(self.idf_log_base)
        return query_text_idf_dict

    @util.timer
    def query_precess(self, query_txts):
        def get_wordnet_pos(tag):
            '''
            This function will get each word speech
            :param tag: have tagged speech of words
            :return:the part of speech of word
            '''
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            else:
                return None

        def clean_str(string):
            string = re.sub(r"[^A-Za-z0-9]", " ", string)
            return string.strip().lower()

        # 词性还原等
        ENGLISH_STOP_WORDS = set(stopwords.words('english'))
        new_query_txts = []
        for query_txt in query_txts:
            new_query_txt = ""
            query_txt = clean_str(query_txt)

            tokens = nltk.word_tokenize(query_txt)
            tagged_sent = pos_tag(tokens)
            wnl = WordNetLemmatizer()
            for tag in tagged_sent:
                wordnet_pos = get_wordnet_pos(tag[1])
                # if tag[1] == 'CD':
                #     wordnet_pos = 'n'
                if wordnet_pos is None:
                    continue
                try:
                    w = wnl.lemmatize(tag[0], pos=wordnet_pos)
                except UnicodeDecodeError:
                    print('%s encoding error' % tag[0])
                    continue
                if w in ENGLISH_STOP_WORDS:
                    continue
                new_query_txt = new_query_txt + " " + w
            new_query_txts.append(new_query_txt)

        return new_query_txts

    def get_query_concept_matrix(self):
        query_concept_matrix = np.zeros((len(self.query_list), len(self.concept_ids)))
        for i, query_txt in enumerate(self.query_list):
            for concept_index, concept in enumerate(self.concept_ids):
                if concept in query_txt:
                    query_concept_matrix[i, concept_index] = 1
                    # print(concept)

        return query_concept_matrix

    def get_query_concept_sim_matrix(self):

        query_concept_matrix = self.query_concept_matrix  # (query_num, concept_num)
        video_concept_array = self.video_concept_array  # (video_num, concept_num)
        model_sim_matrix = self.model_sim_matrix

        # ********得到 concept 相似度**************
        # query_video_sim_matrix = query_concept_matrix.dot(video_concept_array.T)  # (query_num, video_num)
        query_video_sim_matrix = loss.cosine_sim(torch.Tensor(query_concept_matrix), torch.Tensor(video_concept_array))  # (query_num, video_num)
        query_video_sim_matrix = query_video_sim_matrix.numpy()
        # query_video_sim_matrix 对 topK 后面相似度置零
        query_topK_video_index_dict = self.query_topK_video_index_dict
        query_video_sim_matrix_zero = np.zeros(query_video_sim_matrix.shape)
        for query_index in range(query_video_sim_matrix.shape[0]):
            query_video_sim_matrix_zero[query_index, query_topK_video_index_dict[query_index]] = \
                query_video_sim_matrix[query_index, query_topK_video_index_dict[query_index]]

        return query_video_sim_matrix_zero


def process_query():
    # 还没写完
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    video_concept_pkl_path = '/data/liupengju/hf/v3c1/tv19_20_21_a-photo-of-concept_txt2video_sim_matrix.pkl'
    video_id_list = [0, 1, 2, 3, 4]
    model_sim_matrix = np.array(torch.randn((10, 5)))
    query_txts = ["sailboats in the water", "a woman sitting on the floor"]
    re_ranking = Concept_re_ranking(video_concept_pkl_path, video_id_list, model_sim_matrix, query_txts, topK=3)

    query_path = '/data/liupengju/hf/v3c1/TextData/tv21.avs.txt'
    query_dict = {}
    with open(query_path, 'r') as f_q:
        for each in f_q:
            each = each.strip().split(" ", 1)
            query_dict[each[0]] = each[1]
            query_dict[each[0]] = re_ranking.query_precess([each[1]])[0]

    query_expand_dict = {}
    query_repeat_num = 20
    for i, each in enumerate(query_dict):
        query_text_idf_dict = re_ranking.get_idf(query_dict[each])
        for j in range(query_repeat_num):
            query_expand_dict[each + '#' + str(j)] = query_dict[each]
            for concept in query_text_idf_dict:
                query_text_idf_dict[concept]
            np.random.choice()
            query_expand_dict[each+'#'+str(j)] = 1
            pass



if __name__ == "__main__":
    process_query()
    # video_concept_pkl_path = '/data/liupengju/hf/v3c1/tv19_20_21_a-photo-of-concept_txt2video_sim_matrix.pkl'
    # video_id_list = [0, 1, 2, 3, 4]
    # model_sim_matrix = np.array(torch.randn((10, 5)))
    # query_txts = ["sailboats in the water", "a woman sitting on the floor"]
    # re_ranking = Concept_re_ranking(video_concept_pkl_path, video_id_list, model_sim_matrix, query_txts, topK=3)
    # re_ranking.get_query_concept_sim_matrix()
    #
    # q_g_dist = np.array(torch.randn((10, 100)))
    # q_q_dist = np.array(torch.randn((10, 10)))
    # g_g_dist = np.array(torch.randn((100, 100)))
    # re_ranking_tkb_simple(q_g_dist, q_q_dist, g_g_dist, topK=30)