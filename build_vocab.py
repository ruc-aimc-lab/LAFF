#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Pytorch W2VV++
# Written by Xirong Li & Chaoxi Xu
# --------------------------------------------------------



import pickle
from collections import Counter
import os
import sys
import re

from common import *
from util import checkToSkip, makedirsforfile
from textlib import TextTool, Vocabulary


def read_from_txt_file(cap_file):
    """
    从 id string 格式文档中读取 string
    :param cap_file:
    :return:
    """
    captions = []
    with open(cap_file, 'r') as fr:
        for line in fr:
            if len(line.strip().split(' ', 1)) < 2:
                cap_id = line.strip().split(' ', 1)[0]
                caption = ''
            else:
                cap_id, caption = line.split(' ', 1)
            caption = re.sub('#\d\.\d+', "", caption)
            captions.append(caption.strip())
    return captions

def build_vocab(cap_file, encoding, threshold, lang):
    """
    预处理，返回单词 index，以及 bow 字典
    :param cap_file:
    :param encoding: 词编码方法，bow, w2v, gru 等
    :param threshold:
    :param lang:
    :return:
    """
    nosw = '_nsw' in encoding
    logger.info("Build a simple vocabulary wrapper from %s", cap_file)
    captions = read_from_txt_file(cap_file)  # 把其中的词都提取出来
    counter = Counter()
    
    for i, caption in enumerate(captions):
        tokens = TextTool.tokenize(caption, language=lang, remove_stopword=nosw)
        counter.update(tokens)
  
    # Discard if the occurrence of the word is less than min_word_cnt.
    word_counts = [(word,cnt) for word, cnt in list(counter.items()) if cnt >= threshold]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary(encoding)
    if 'gru' in encoding:
        vocab.add('<pad>')
        vocab.add('<start>')
        vocab.add('<end>')
        vocab.add('<unk>')

    # Add words to the vocabulary.
    for word, c in word_counts:
        vocab.add(word)
    return vocab, word_counts


def process(options, collection):
    overwrite = options.overwrite
    rootpath = options.rootpath
    threshold = options.threshold
    encoding = options.encoding
    language = options.language
    folder_name = options.folder_name
    caption_name = options.caption_name

    vocab_file = os.path.join(rootpath, collection, 'TextData', folder_name, '%s_%d.pkl'% (encoding, threshold))
    count_file = os.path.join(os.path.dirname(vocab_file), '%s_%d.txt'% (encoding, threshold))

    if checkToSkip(vocab_file, overwrite):
        return 0
    
    cap_file = os.path.join(rootpath, collection, 'TextData', caption_name)  # 这个是输入的caption文件
    vocab, word_counts = build_vocab(cap_file, encoding, threshold=threshold, lang=language)
    
    makedirsforfile(vocab_file)
    with open(vocab_file, 'wb') as fw:
        pickle.dump(vocab, fw, pickle.HIGHEST_PROTOCOL)
    logger.info("Saved vocabulary of %d words to %s", len(vocab), vocab_file)

    with open(count_file, 'w') as fw:
        fw.write('\n'.join(['%s %d' % x for x in word_counts]))
    
    logger.info("Saved word-counts to %s", count_file)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
        
    from optparse import OptionParser
 
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--caption_name", default='train_collection.caption.txt', type="string", help="caption_name")
    parser.add_option("--language", default=DEFAULT_LANG, type="string", help="language (default: %s)" % DEFAULT_LANG)
    parser.add_option("--encoding", default="bow", type="choice", choices=TEXT_ENCODINGS, help="text encoding strategy. Valid choices are %s. (default: %s)" % (TEXT_ENCODINGS, DEFAULT_TEXT_ENCODING))
    parser.add_option("--threshold", default=5, type="int", help="minimum word occurrence (default: %d)" % MIN_WORD_COUNT)
    parser.add_option("--folder_name", default="vocab", type="string", help="The output folder name (default: vocab)" )
                      
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    assert(options.language in ['en', 'zh']), 'language %s not supported' % options.language
    
    return process(options, args[0])

if __name__=="__main__":
    if len(sys.argv) == 1:
        sys.argv = "build_vocab.py msrvtt10k_object_labels --overwrite 0 " \
                   "--rootpath /home/liupengju/hf_code/VisualSearch " \
                   "--caption_name msrvtt10k.caption.vg_consecutive_2_frames_label-pred.txt " \
                   "--encoding bow --folder_name vocab_vg_consecutive_2_frames_label-pred".split(' ')

    sys.exit(main())