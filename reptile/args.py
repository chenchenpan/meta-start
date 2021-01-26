"""
Command-line argument parsing.
"""

import argparse
from functools import partial

import tensorflow as tf

from .reptile import Reptile, FOML


def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--data_path', help='path to dataset.',
                        type=str)
    parser.add_argument('--output_dir', help='Folder to store outputs.',
                        type=str)
    parser.add_argument('--n_features', help='number of features in NN.', default=6, type=int)
    parser.add_argument('--n_layers', help='number of hidden layers in NN.', default=1, type=int)
    parser.add_argument('--hidden_size', help='number of hidden units in NN.', default=32, type=int)
    parser.add_argument('--num_repeats', help='number of repeats of experiments.', default=5, type=int)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--num_shots', help='number of examples per class', default=5, type=int)

    parser.add_argument('--num_test_shots', help='number of test per class', default=2, type=int)

    parser.add_argument('--num_train_shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument('--inner_iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--learning_rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta_step_size', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta_step_size_final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument('--meta_batch_size', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta_iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval_inner_iters', help='eval inner iterations', default=20, type=int)
    parser.add_argument('--num_eval_samples', help='evaluation samples', default=100, type=int)
    parser.add_argument('--eval_interval', help='train steps per eval', default=10, type=int)
    parser.add_argument('--weight_decay_rate', help='weight decay rate', default=1, type=float)
    parser.add_argument('--foml', help='use FOML instead of Reptile', action='store_true')
    parser.add_argument('--foml_tail', help='number of shots for the final mini-batch in FOML',
                        default=None, type=int)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true',
                        default=False)
    parser.add_argument('--use_hypernet', help='whether to use hypernet or not.', action='store_true',
                        default=False)
    parser.add_argument('--use_cat_embed', help='whether to use category embedding or not.', action='store_true',
                        default=False)
    parser.add_argument('--cat_embed_path', help='path to category embeddings.',
                        type=str)
    return parser

