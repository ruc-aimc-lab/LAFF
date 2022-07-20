# coding=utf-8
import os, sys, array
import time

import numpy as np
import torch

import util
from itertools import tee
import multiprocessing as mp


class BigFile:

    def __init__(self, datadir, bin_file="feature.bin"):
        self.nr_of_images, self.ndims = list(map(int, open(os.path.join(datadir, 'shape.txt')).readline().split()))
        id_file = os.path.join(datadir, "id.txt")
        self.names = open(id_file, 'r').read().strip().split('\n')  # 所有 video 文件名
        if len(self.names) != self.nr_of_images:
            self.names = open(id_file, 'r').read().strip().split(' ')
        assert(len(self.names) == self.nr_of_images)
        self.name2index = dict(list(zip(self.names, list(range(self.nr_of_images)))))  # 给每一个文件名弄一个编号
        self.binary_file = os.path.join(datadir, bin_file)
        print(("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir)))

        # split the file and accelerate
        # offset = np.float32(1).nbytes * self.ndims
        # split_num = 10
        # if self.nr_of_images < split_num * 5:
        #     self.fr_list = None
        #     return
        # self.segmentation = int(self.nr_of_images / split_num)
        # self.fr_list = []
        # for each in range(split_num-1):
        #     fr_list = [open(self.binary_file, 'rb') for each_1 in range(5)]
        #     [each_1.seek(each*self.segmentation*offset, 0) for each_1 in fr_list]
        #     self.fr_list.append([{'offset': fr.tell(), 'fr': fr} for fr in fr_list])
        # self.mp_signal = mp.Array('i', [1] * len(self.fr_list)*len(self.fr_list[0]))

        # method 2, read all the file and store it
        self.torch_array = None

    def read_all_and_store(self):
        def readall(self, ndims):
            torch_array = torch.zeros(ndims, dtype=torch.half)

            index_name_array = [(self.name2index[x], x) for x in set(self.names) if x in self.name2index]

            index_name_array.sort(key=lambda v: v[0])
            sorted_index = [x[0] for x in index_name_array]

            nr_of_images = len(index_name_array)

            offset = np.float32(1).nbytes * self.ndims

            res1 = array.array('f')
            fr = open(self.binary_file, 'rb')
            fr.seek(index_name_array[0][0] * offset)
            res1.fromfile(fr, self.ndims)
            previous = index_name_array[0][0]
            torch_array[previous] = torch.tensor(res1)

            for next in sorted_index[1:]:
                res1 = array.array('f')
                move = (next - 1 - previous) * offset
                # print next, move
                fr.seek(move, 1)
                res1.fromfile(fr, self.ndims)
                previous = next
                torch_array[previous] = torch.tensor(res1)

            return torch_array
        self.torch_array = readall(self, self.shape())

    def readall(self, isname=True):
        index_name_array = [(self.name2index[x], x) for x in set(self.names) if x in self.name2index]

        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims
        
        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]
 
        for next in sorted_index[1:]:
            move = (next-1-previous) * offset
            #print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next


        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]

    def _read_from_ram(self, requested, isname=True):
        """
        从内存中直接读
        :param requested:
        :param isname:
        :return: 这里主要是视频名字和 feature vector, 一般输出 list
        """
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        if self.torch_array is None:
            self.read_all_and_store()

        res = self.torch_array[index_name_array[0][0]]

        # print([ res.tolist() ])
        # print(self.read(requested)[1])

        return [index_name_array[0][1]], [ res.tolist() ]

    def _read_one_(self, requested, isname=True):
        """
        根据文件名读取文件，具体是从 bin 文件中读取numpy 矩阵，这里主要是视频名字和 feature vector
        :param requested:
        :param isname:
        :return: 这里主要是视频名字和 feature vector, 一般输出 list
        """
        if self.fr_list is None:
            return self.read(requested, isname)
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')

        try:
            index = int(index_name_array[0][0] / self.segmentation)
            if index >= len(self.fr_list):
                index = len(self.fr_list)-1

            # 获取信号量
            signal = True
            while signal:
                with self.mp_signal.get_lock():  # 直接调用get_lock()函数获取锁
                    for signal_index in range(len(self.fr_list[index])):
                        if self.mp_signal[index*len(self.fr_list[0]) + signal_index] == 1:
                            self.mp_signal[index*len(self.fr_list[0]) + signal_index] = 0
                            signal = False
                            break
                if signal:
                    time.sleep(0.0001)

            fr = self.fr_list[index][signal_index]['fr']
            move = index_name_array[0][0] * offset - fr.tell()
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            fr.seek(-move - offset, 1)
            self.mp_signal[index*len(self.fr_list[0]) + signal_index] = 1

            # with open(self.binary_file, 'rb') as fr:
            #     move = index_name_array[0][0] * offset
            #     fr.seek(move)
            #     res.fromfile(fr, self.ndims)

        except Exception as e:
            print(e)

        # print([ res.tolist() ])
        # print(self.read(requested)[1])

        return [index_name_array[0][1]], [ res.tolist() ]

    def read(self, requested, isname=True):
        """
        根据文件名读取文件，具体是从 bin 文件中读取numpy 矩阵，这里主要是视频名字和 feature vector
        :param requested: []
        :param isname:
        :return: 这里主要是视频名字和 feature vector
        """
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')

        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]

        for next in sorted_index[1:]:
            move = (next-1-previous) * offset
            #print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]


    def read_one(self, name):
        # renamed, vectors = self._read_one_([name])
        renamed, vectors = self.read([name])
        # renamed, vectors = self._read_from_ram([name])
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]

    def cal_time(self):
        pass

        
class StreamFile:

    def __init__(self, datadir):
        self.feat_dir = datadir
        self.nr_of_images, self.ndims = list(map(int, open(os.path.join(datadir,'shape.txt')).readline().split()))
        id_file = os.path.join(datadir, "id.txt")
        self.names = open(id_file, 'r').read().strip().split('\n')  # 所有 video 文件名
        if len(self.names) != self.nr_of_images:
            self.names = open(id_file, 'r').read().strip().split(' ')
        assert(len(self.names) == self.nr_of_images)
        self.name2index = dict(list(zip(self.names, list(range(self.nr_of_images)))))
        self.binary_file = os.path.join(datadir, "feature.bin")
        print(("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir)))
        self.fr = None
        self.current = 0
    

    def open(self):
        self.fr = open(os.path.join(self.feat_dir,'feature.bin'), 'rb')
        self.current = 0

    def close(self):
        if self.fr:
            self.fr.close()
            self.fr = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current >= self.nr_of_images:
            self.close()
            raise StopIteration
        else:
            res = array.array('f')
            res.fromfile(self.fr, self.ndims)
            _id = self.names[self.current]
            self.current += 1
            return _id, res.tolist() 
            

if __name__ == '__main__':
    feat_dir = "/data2/hf/VisualSearch/toydata/FeatureData/f1"
    bigfile = BigFile(feat_dir)

    imset = str.split('b z a a b c')
    renamed, vectors = bigfile.read(imset)


    for name,vec in zip(renamed, vectors):
        print(name, vec)
        
    bigfile = StreamFile(feat_dir)
    bigfile.open()
    for name, vec in bigfile:
        print(name, vec)
    bigfile.close()

    
        
