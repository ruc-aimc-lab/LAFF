# coding=utf-8
import os
import sys
import random
import numpy as np
import unittest

from common import ROOT_PATH
rootpath = ROOT_PATH

train_collection ='tgif-msrvtt10k'
val_collection = 'tv2016train'
test_collection = 'iacc.3'

#vis_feat = 'mean_resnext101_resnet152'
vis_feat = 'mean_pyresnext-101_rbps13k,flatten0_output,os'


class TestSuite (unittest.TestCase):

    def test_rootpath(self):
        self.assertTrue(os.path.exists(rootpath))

    def test_w2v_dir(self):
        w2v_dir = os.path.join(rootpath, 'word2vec/flickr/vec500flickr30m')
        self.assertTrue(os.path.exists(w2v_dir), 'missing %s'%w2v_dir)

    def test_train_data(self):
        cap_file = os.path.join(rootpath, train_collection, 'TextData', '%s.caption.txt' % train_collection)
        self.assertTrue(os.path.exists(cap_file), 'missing %s'%cap_file)
        feat_dir = os.path.join(rootpath, train_collection, 'FeatureData', vis_feat)
        self.assertTrue(os.path.exists(feat_dir), 'missing %s'%feat_dir)

    def test_val_data(self):
        cap_file = os.path.join(rootpath, val_collection, 'TextData', '%s.caption.txt' % val_collection)
        self.assertTrue(os.path.exists(cap_file), 'missing %s'%cap_file)
        feat_dir = os.path.join(rootpath, val_collection, 'FeatureData', vis_feat)
        self.assertTrue(os.path.exists(feat_dir), 'missing %s'%feat_dir)

    def test_test_data(self):
        for test_set in 'tv16 tv17 tv18'.split():
            topic_file = os.path.join(rootpath, test_collection, 'TextData', '%s.avs.txt' % test_set)
            self.assertTrue(os.path.exists(topic_file), 'missing %s'%topic_file)  
            
            gt_file = os.path.join(rootpath, test_collection, 'TextData', 'avs.qrels.%s' % test_set)
            self.assertTrue(os.path.exists(gt_file), 'missing %s'%gt_file)
            
        feat_dir = os.path.join(rootpath, test_collection, 'FeatureData', vis_feat)
        self.assertTrue(os.path.exists(feat_dir), 'missing %s'%feat_dir)
        

suite = unittest.TestLoader().loadTestsFromTestCase(TestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)

