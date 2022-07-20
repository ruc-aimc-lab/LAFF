#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Pytorch W2VV++
# Written by Xirong Li & Chaoxi Xu
# --------------------------------------------------------


import os
import logging
import torch

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
MIN_WORD_COUNT = 5

TEXT_ENCODINGS = ['bow', 'bow_nsw', 'gru']
DEFAULT_TEXT_ENCODING = 'bow'
DEFAULT_LANG = 'en'

logger = logging.getLogger(__file__)  # 给 looger 命名为当前文件名
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)

torch.multiprocessing.set_sharing_strategy('file_system')  # 多线程

class No:
    pass

