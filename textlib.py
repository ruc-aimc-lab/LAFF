# -*- coding: utf-8 -*-
import os
import re
import sys
import numpy as np

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)

en_stop_fname = os.path.join(os.path.dirname(__file__), 'stopwords_en.txt')
zh_stop_fname = os.path.join(os.path.dirname(__file__), 'stopwords_zh.txt')
ENGLISH_STOP_WORDS = set(map(str.strip, open(en_stop_fname).readlines()))
CHINESE_STOP_WORDS = set(map(str.strip, open(zh_stop_fname).readlines()))

if 3 == sys.version_info[0]:  # python 3
    CHN_DEL_SET = '， 。 、 ！ 《 》 “ ” ； ？ ‘ ’ '.split()
else:
    CHN_DEL_SET = [x.decode('utf-8') for x in '， 。 、 ！ 《 》 “ ” ； ？ ‘ ’ '.split()]

class TextTool:
    @staticmethod
    def tokenize(input_str, clean=True, language='en', remove_stopword=False):
        """
        进行预处理，返回一个 list
        :param input_str:
        :param clean: 如果 true，去掉不是英文字母和数字的。
        :param language:
        :param remove_stopword: 如果 true，去掉 stopword
        :return:
        """
        if 'en' == language:  # English
            # delete non-ascii chars
            #sent = input_str.decode('utf-8').encode('ascii', 'ignore')
            sent = input_str
            if clean:
                sent = sent.replace('\r',' ')
                sent = re.sub(r"[^A-Za-z0-9]", " ", sent).strip().lower()
            tokens = sent.split()
            if remove_stopword:
                tokens = [x for x in tokens if x not in ENGLISH_STOP_WORDS]
        else: # Chinese  
        # sent = input_str #string.decode('utf-8')
            sent = input_str.decode('utf-8')
            
            if clean:
                for elem in CHN_DEL_SET:
                    sent = sent.replace(elem,'')
            sent = sent.encode('utf-8')
            sent = re.sub("[A-Za-z]", "", sent)
            tokens = [x for x in sent.split()] 
            if remove_stopword:
                tokens = [x for x in tokens if x not in CHINESE_STOP_WORDS]

        return tokens
def negation_augumentation(input_str):
    res = [input_str]
    replacelist = [("don t", "do not"), ("doesn t", "does not"), ("didn t", "did not"), ("isn t", "is not"),
                   ("aren t", "are not"), ("wasn t", "was not"), ("weren t", "were not"),
                   ("won t", "will not"), ("hasn t", "has not"), ("haven t", "have not"), ("can t", "can not"),
                   ("couldn t", "could not"),
                   ("don't", "do not"), ("doesn't", "does not"), ("didn't", "did not"), ("isn't", "is not"),
                   ("aren't", "are not"), ("won't", "will not"), ("hasn't", "has not"), ("haven't", "have not"),
                   ("can't", "can not"), ("couldn't", "could not")]
    for pairs in replacelist:
        if input_str.find(pairs[0]) != -1:
            input_str2 = re.sub(pairs[0], pairs[1], input_str)
            res .append(input_str2)
            break
    for pairs in replacelist:
        if input_str.find(pairs[1]) != -1:
            input_str2 = re.sub(pairs[1], pairs[0], input_str)
            res.append(input_str2)
            break
    return res

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, encoding):
        self.word2idx = {}
        self.idx2word = {}
        self.encoding = encoding

    def add(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def find(self, word):
        return self.word2idx.get(word, -1)  # word to index
    
    
    def __getitem__(self, index):
        return self.idx2word[index]
 
    def __call__(self, word):
        if (word not in self.word2idx):
            if 'gru' in self.encoding:
                return self.word2idx['<unk>']
            else:
                raise Exception ('word out of vocab: %s' % word)
        else:
            return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    
if __name__ == '__main__':
    test_strs = '''a Dog??? is running
The dog runs
dogs-x runs'''.split('\n')

    for t in test_strs:
        print(t, '->', TextTool.tokenize(t, clean=True, language='en'), '->', TextTool.tokenize(t, 'en', True))
        
    test_strs = '''一间 干净 整洁 的 房间 。
一只 黄色 的 小dsc狗csd 趴在 长椅 上dcdcqdeded'''.split('\n')
    
    for t in test_strs:
        print(t, '->', ' '.join(TextTool.tokenize(t, clean=True, language='zh')), '->', ' '.join(TextTool.tokenize(t, 'zh', True)))
    
