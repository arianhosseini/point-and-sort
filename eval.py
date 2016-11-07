#!/usr/bin/env python

import logging
import numpy as np
import sys
import os
import importlib
import cPickle

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

import data
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

class EvaluateSorterModel(SimpleExtension):
    def __init__(self, path, model, data_stream, vocab_size, vocab, eval_mode, quiet=False, iter_num=1, **kwargs):
        super(EvaluateSorterModel, self).__init__(**kwargs)
        self.path = path
        self.model = model
        self.data_stream = data_stream
        self.iter_num = iter_num
        self.gen_fun = self.model.get_theano_function()
        self.vocab_size = vocab_size
        self.eval_mode = eval_mode
        self.quiet = quiet
        self.vocab = vocab
        self.best_macroF1 = 0.0

    def compute_batch(self, data, batch_num):
        answer = data['answer']
        data.pop('answer_mask', None)
        data.pop('answer', None)
        temp = self.gen_fun(**data)
        predictions_indices = np.asarray(temp).T
        unsorted = data['unsorted']
        print unsorted
        predictions = []
        for i,ctx in enumerate(unsorted): #changing indices to actual words (in this case, numbers)
            predictions.append(unsorted[i,predictions_indices[i]])

        predictions = np.asarray(predictions)
        print "predictions:"
        print predictions

        if self.quiet==False:
            for i in range(len(predictions)):
                for j in range(len(predictions[i,:])):
                    if self.vocab[predictions[i,j]] == "<EOA>":
                        print "found <EOA>, all zeros"
                        predictions[i,j:] = 0
                        break;
            for i in range(len(predictions)):
                print "answer:",
                for j in range(len(answer[i,:])):
                    print self.vocab[unsorted[i,answer[i,j]]],
                print ""
                print "predictions: ",
                for j in range(len(predictions[i,:])):
                    if predictions[i,j] > 1:
                        print self.vocab[predictions[i,j]],
                print ""
                print "unsorted: ",
                for j in range(len(unsorted[i,:])):
                    if unsorted[i, j] > 1 :
                        print self.vocab[unsorted[i, j]],

                print
                print ""


        unsorted = (unsorted[:,:,None] == np.arange(self.vocab_size)).sum(axis = 1).clip(0,1)
        answer_bag = (answer[:,None] == np.arange(self.vocab_size)[:,None]).sum(axis=2).clip(0,1)
        answer_bag[:,0] = 0
        predictions_bag = (predictions[:,None] == np.arange(self.vocab_size)).sum(axis=1).clip(0,1).sum(axis=1)
        predictions_bag[:,0] = 0

        selected_items = predictions_bag.sum(axis=1, dtype=float)
        precision = np.zeros(shape=(selected_items.shape[0]),dtype=float)
        recall = np.zeros(shape=(selected_items.shape[0]),dtype=float)
        macroF1 = np.zeros(shape=(selected_items.shape[0]),dtype=float)
        answers_bag = []

        num_of_examples = selected_items.shape[0]
        precision_sum = precision.sum()
        recall_sum = recall.sum()
        f1_sum = macroF1.sum()
        exact = (precision * recall == 1)
        exact_sum = exact.sum()

        avg_precision = precision.mean()
        avg_recall = recall.mean()
        macroF1_of_avg =  (2 * ( avg_precision * avg_recall )) / (avg_precision + avg_recall)

        return (precision_sum, recall_sum, exact_sum, f1_sum, num_of_examples)

    def do_load(self):
        try:
            with open(self.path, 'r') as f:
                print 'Loading parameters from ' + self.path
                self.model.set_parameter_values(cPickle.load(f))
        except IOError:
            print 'Error in loading!'

    def do(self, which_callback, *args):
        if self.path != "":
            self.do_load()
        epoch_iter = self.data_stream.get_epoch_iterator(as_dict=True)

        count = 0
        macroF1 = 0.0
        num_of_examples = 0.0
        precision_sum, recall_sum, exact_sum,f1_sum = 0.0 , 0.0 , 0.0, 0.0

        for data in epoch_iter:
            # data = epoch_iter.next()

            # print('batch %d'%count)
            count += 1
            p,r,e,f1,n = self.compute_batch(data, count)
            precision_sum += p
            recall_sum += r
            exact_sum += e
            f1_sum += f1
            num_of_examples += n

            if self.eval_mode == 'batch':
                break

        avg_precision = precision_sum / num_of_examples
        avg_recall = recall_sum / num_of_examples
        avg_exact = exact_sum / num_of_examples
        macroF1 =  (2 * ( avg_precision * avg_recall )) / (avg_precision + avg_recall)
        avg_of_f1s = f1_sum / num_of_examples
        self.best_macroF1 = max(self.best_macroF1, macroF1)
        print('Validation Set:')
        print "         avg_recall: " + str(avg_recall)
        print "         avg_precision: " + str(avg_precision)
        print "         macroF1: " + str(macroF1)
        print "         averageF1: " + str(avg_of_f1s)
        print "         exact match acc: " + str(avg_exact)
        print "         # of examples: " + str(num_of_examples)
        print "         best macroF1: " + str(self.best_macroF1)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    eval_mode = 'batch'
    if len(sys.argv) == 3:
        eval_mode = 'all'

    config = importlib.import_module(model_name)

    # Build datastream
    path = os.path.join(os.getcwd(), "data/data.txt")
    ds, valid_stream = data.setup_sorter_datastream(path,config)
    snapshot_path = os.path.join("model_params", model_name+".pkl")

    # Build model
    m = config.Model(config, ds.vocab_size)

    # Build the Blocks stuff for training
    test_model = Model(m.generations)
    model = Model(m.sgd_cost)
    algorithm = None
    extensions = [EvaluateSorterModel(path=snapshot_path, model=test_model, data_stream=valid_stream, vocab_size = ds.vocab_size, vocab = ds.vocab, eval_mode=eval_mode, before_training=True)]

    main_loop = MainLoop(
        model=model,
        data_stream=valid_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
