#!/usr/bin/env python
import logging
import numpy
import sys
import os
import importlib

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent

try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

import data
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module(model_name)

    path = os.path.join(os.getcwd(), "data/data.txt")
    # valid_path = os.path.join(os.getcwd(), "squad_rare/dev-v1.0_tokenized.json")
    valid_path = None

    ds, train_stream = data.setup_sorter_datastream(path,config)
    dump_path = os.path.join("model_params", model_name+".pkl")
    valid_stream = None

    m = config.Model(config, ds.vocab_size)
    model = Model(m.sgd_cost)
    # test_model = Model(m.generations)
    algorithm = GradientDescent(cost=m.sgd_cost,
                                step_rule=config.step_rule,
                                parameters=model.parameters,
                                on_unused_sources='ignore')

    extensions = [
            TrainingDataMonitoring(
                [v for l in m.monitor_vars for v in l],
                prefix='train',
                after_epoch=True)
    ]
    if config.save_freq is not None and dump_path is not None:
        extensions += [
            SaveLoadParams(path=dump_path,
                           model=model,
                           before_training=False,
                           after_training=True,
                           after_epoch=True)
        ]
    if valid_stream is not None and config.valid_freq != -1:
        extensions += [
            DataStreamMonitoring(
                [v for l in m.monitor_vars_valid for v in l],
                valid_stream,
                prefix='valid'),
        ]

    extensions += [
            Printing(after_epoch=True),
            # EvaluateModel(path="", model=test_model, data_stream=valid_stream, vocab_size = ds.vocab_size, vocab = ds.vocab, eval_mode='batch', quiet=True, after_epoch=True),
            ProgressBar()
    ]

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions
    )

    # Run the model !
    main_loop.run()
    main_loop.profile.report()
