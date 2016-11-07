import logging
import random
import numpy
import pprint
import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme, ShuffledScheme, ShuffledExampleScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os
import json
import itertools

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class SorterDataset(Dataset):
    def __init__(self, path, **kwargs):

        self.provides_sources = ('unsorted','answer')
        self.path = path
        self.data = open(path,'r').readlines() #actual json data
        self.vocab = ['<EOA>'] + [str(i) for i in range(0,2001)]

        self.vocab_size = len(self.vocab)
        # print("vocab size: %d"%self.vocab_size)
        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}
        super(SorterDataset, self).__init__(**kwargs)

    def to_word_id(self, w):
        ''' word to index'''
        if w in self.reverse_vocab:
            return self.reverse_vocab[w]
        else:
            print "out: ", w
            return self.reverse_vocab['<UNK>']

    def to_word_ids(self, s):
        ''' words to indices '''
        return numpy.array([self.to_word_id(x) for x in s], dtype=numpy.int32)


    def get_index(self, l, item, offset=None):
        if offset:
            return offset + 1 + l[offset+1].index(item)
        else:
            return l.index(item)

    def get_data(self, state=None, request=None):
        # print request
        if request is None or state is not None:
            raise ValueError("Expected a request (name of a question file) and no state.")

        unsorted = '<EOA> '+self.data[request*2].strip()
        answer = self.data[request*2+1].strip()+' <EOA>'

        unsorted_ids = self.to_word_ids(unsorted.split(' '))
        a = self.to_word_ids(answer.split(' '))
        ans_indices = []
        for i in a:
            index = numpy.where(unsorted_ids==i)[0]
            count = 0
            while index[count] in ans_indices:
                count += 1
            ans_indices.append(index[count])
        # print "u: ", unsorted_ids
        # print "i: ", ans_indices
        ans_indices = numpy.asarray(ans_indices, dtype=numpy.int32)

        return (unsorted_ids, ans_indices)

# -------------- DATASTREAM SETUP --------------------
class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]

def create_dataset(path='data',name="data.txt"):
    unsorteds = []
    answers = []
    lengths = []
    data_file = open(os.path.join(path,name),'w')
    print "creating ..."
    for i in range(1000000):
        if i % 100000 == 0:
            print i
        length = random.randint(1, 10)
        lengths.append(length)
        lower_bound = random.randint(0,500)
        upper_bound = random.randint(lower_bound,2000)
        unsorted = [random.randint(lower_bound,upper_bound) for _ in range(length)]
        unsorteds.append(unsorted)
        answer = sorted(unsorted)
        answers.append(answer)
        unsorted_str = ' '.join(map(str,unsorted))
        answer_str = ' '.join(map(str,answer))
        data_file.write(unsorted_str+'\n')
        data_file.write(answer_str+'\n')
    data_file.close()
    print "done"

def setup_sorter_datastream(path, config):
    ds = SorterDataset(path)
    it = ShuffledExampleScheme(examples=config.example_count)
    stream = DataStream(ds, iteration_scheme=it)
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size * config.sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('unsorted'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))
    stream = Padding(stream, mask_sources=['answer','unsorted'], mask_dtype='int32')
    return ds, stream


if __name__ == "__main__":
    class DummyConfig:
        def __init__(self):
            self.batch_size = 2
            self.sort_batch_count = 1000

    ds, stream = setup_sorter_datastream("data/data.txt",DummyConfig())
    it = stream.get_epoch_iterator()
    for i, d in enumerate(stream.get_epoch_iterator()):
        print '--'
        print d
