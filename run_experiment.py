#!/usr/bin/env python
# coding: utf-8

import json
import os
import numpy as np
from collections import Counter
from functools import partial

import sklearn
from sklearn import preprocessing
import reptile
import reptile.model
import reptile.train
import reptile.eval
import reptile.reptile
import tensorflow as tf
import reptile.args


def main():
    args = reptile.args.argument_parser().parse_args()
    
    np.random.seed(args.seed)

    if args.use_cat_embed:
        with open(args.cat_embed_path, 'r') as f:
            cat_embeddings = json.load(f)

        for k, v in cat_embeddings.items():
            cat_embeddings[k] = np.array(v)

        cat_embed_dim = cat_embeddings[k].shape[0]
    else:
        cat_embeddings = None
        cat_embed_dim = 0

    with open(args.data_path, 'r') as f:
        splitted_data = json.load(f)

    print('{} total unique companies.'.format(len(splitted_data['companies'])))

    company_int_keys = sorted([int(s) for s in splitted_data['companies'].keys()])

    company_col_dict = {}
    company_int_key_to_pos = {}
    for k in splitted_data['companies']['1'].keys():
        company_col_dict[k] = []
    for i, key in enumerate(company_int_keys):
        company_info = splitted_data['companies'][str(key)]
        for col, v in company_info.items():
            company_col_dict[col].append(v)
            company_int_key_to_pos[key] = i

    feature_names = []
    col_arrays = []
    for k, v in company_col_dict.items():
        if not isinstance(v[0], str) and k != 'label':
            feature_names.append(k)
            col_arrays.append(np.array(company_col_dict[k]))
    company_mat = np.vstack(col_arrays).transpose()
    raw_data_mat = company_mat
    label_mat = np.array(company_col_dict['label'])

    print('feature_names: {}'.format(feature_names))
    print('raw data matrix shape: {}'.format(raw_data_mat.shape))
    print('label matrix shape: {}'.format(label_mat.shape))

    def update_to_new_id(old_dict):
        new_dict = {}
        for k, v in old_dict.items():
            new_dict[k] = [company_int_key_to_pos[idx] for idx in old_dict[k]]
        return new_dict

    train_cat_dict = update_to_new_id(splitted_data['train'])
    dev_cat_dict = update_to_new_id(splitted_data['dev'])
    test_cat_dict = update_to_new_id(splitted_data['test'])

    def collect_ids(cat_dict):
        total_ids = set()
        for k, v in cat_dict.items():
            total_ids = total_ids.union(set(v))
        return total_ids

    train_ids = collect_ids(train_cat_dict)
    print('train: {} cats, {} unique companies, positive ratio: {}'.format(
        len(train_cat_dict.keys()), len(train_ids), np.mean(label_mat[list(train_ids)])))
    dev_ids = collect_ids(dev_cat_dict)
    print('dev: {} cats, {} unique companies, positive ratio: {}'.format(
        len(dev_cat_dict.keys()), len(dev_ids), np.mean(label_mat[list(dev_ids)])))
    test_ids = collect_ids(test_cat_dict)
    print('test: {} cats, {} unique companies, positive ratio: {}'.format(
        len(test_cat_dict.keys()), len(test_ids), np.mean(label_mat[list(test_ids)])))

    print('Compare to check whether id update is correct:')
    print(splitted_data['companies']['29337'])
    print(company_mat[company_int_key_to_pos[29337]])

    train_data = raw_data_mat[list(train_ids)]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    data_mat = scaler.transform(raw_data_mat)
    train_set = Dataset(train_cat_dict, data_mat, label_mat, cat_embeddings=cat_embeddings)
    dev_set = Dataset(dev_cat_dict, data_mat, label_mat, cat_embeddings=cat_embeddings)
    test_set = Dataset(test_cat_dict, data_mat, label_mat, cat_embeddings=cat_embeddings)

    print('Example mini dataset:\n{}'.format(train_set.sample_mini_dataset(args.num_shots)))

    test_accs = []
    train_accs = []
    if args.foml:
        reptile_fn = partial(reptile.reptile.FOML, tail_shots=args.foml_tail)
    else:
        reptile_fn = reptile.reptile.Reptile

    for i in range(args.num_repeats):
        tf.reset_default_graph()

        if args.use_hypernet:
            assert args.use_cat_embed
            model = reptile.model.HyperNNClassifier(
                2, n_features=args.n_features+cat_embed_dim,
                n_layers=args.n_layers,
                task_descr_dim=cat_embed_dim,
                hidden_size=args.hidden_size)
        else:
            model = reptile.model.NNClassifier(
                2, n_features=args.n_features+cat_embed_dim,
                n_layers=args.n_layers, hidden_size=args.hidden_size)
        
        with tf.Session() as sess:
            print('=' * 25 + '  repeat {}  '.format(i) + '=' * 25)
            print('Training...')
            reptile.train.train(
                sess, model, train_set, test_set, args.output_dir+'_{}'.format(i),
                meta_iters=args.meta_iters, meta_step_size=args.meta_step_size,
                meta_step_size_final=args.meta_step_size_final,
                meta_batch_size=args.meta_batch_size,
                inner_iters=args.inner_iters, num_shots=args.num_shots,
                num_train_shots=args.num_train_shots,
                eval_interval=args.eval_interval, weight_decay_rate=args.weight_decay_rate)
            print('Evaluating...')
            train_acc = reptile.eval.evaluate(
                sess, model, train_set, num_samples=args.num_eval_samples,
                eval_inner_iters=args.eval_inner_iters, num_shots=args.num_shots,
                weight_decay_rate=args.weight_decay_rate)
            print('Train accuracy: ' + str(train_acc))
            test_acc = reptile.eval.evaluate(
                sess, model, test_set, num_samples=args.num_eval_samples,
                eval_inner_iters=args.eval_inner_iters, num_shots=args.num_shots,
                weight_decay_rate=args.weight_decay_rate)
            print('Test accuracy: ' + str(test_acc))
            train_accs.append(train_acc)
            test_accs.append(test_acc)

    print('train accuracies: {}'.format(train_accs))
    print('test accuracies: {}'.format(test_accs))
    mean_train_acc = np.mean(train_accs) * 100
    std_train_acc = np.std(train_accs) * 100
    print('average train accuracy: {0:.5f}+-{1:.5f}%'.format(mean_train_acc, std_train_acc))
    mean_test_acc = np.mean(test_accs) * 100
    std_test_acc = np.std(test_accs) * 100
    print('average test accuracy: {0:.5f}+-{1:.5f}%'.format(mean_test_acc, std_test_acc))


class Dataset:
    def __init__(self, cat_dict, data_mat, label_mat, cat_embeddings=None):
        self.cat_dict = cat_dict
        self.data_mat = data_mat
        self.label_mat = label_mat
        self.cat_embeddings = cat_embeddings

    def sample_mini_dataset(self, n_example_per_class):
        
        all_cats = list(self.cat_dict.keys())
        filtered_cats = []
        for cat in all_cats:
            all_ids = self.cat_dict[cat]
            pos_ids = [x for x in all_ids if self.label_mat[x] == 1]
            neg_ids = [x for x in all_ids if self.label_mat[x] == 0]
            if len(pos_ids) >= n_example_per_class and len(neg_ids) >= n_example_per_class:
                filtered_cats.append(cat)
        
#         cats = list(self.cat_dict.keys())
        selected_cat = np.random.choice(filtered_cats)
        
        print('*'*20)
        print(selected_cat)
        print('*'*20)
        
        all_ids = self.cat_dict[selected_cat]
        pos_ids = [x for x in all_ids if self.label_mat[x] == 1]
        neg_ids = [x for x in all_ids if self.label_mat[x] == 0]
#         if len(pos_ids) < n_example_per_class:
#             print(n_example_per_class)
#             print(pos_ids)
#         if len(neg_ids) < n_example_per_class:
#             print(neg_ids)
        selected_pos_ids = np.random.choice(pos_ids, n_example_per_class, replace=False)
        selected_neg_ids = np.random.choice(neg_ids, n_example_per_class, replace=False)
        pos_mat = self.data_mat[selected_pos_ids]
        neg_mat = self.data_mat[selected_neg_ids]
        if self.cat_embeddings:
            n_pos = len(selected_pos_ids)
            n_neg = len(selected_neg_ids)
            embed_dim = self.cat_embeddings[selected_cat].shape[0]
            cat_vec = np.reshape(self.cat_embeddings[selected_cat], (1, embed_dim))
            pos_mat = np.concatenate(
                [pos_mat, np.tile(cat_vec, [n_pos, 1])], axis=1)
            neg_mat = np.concatenate(
                [neg_mat, np.tile(cat_vec, [n_neg, 1])], axis=1)
        return (list(zip(pos_mat, self.label_mat[selected_pos_ids])) +
                list(zip(neg_mat, self.label_mat[selected_neg_ids])))
    

if __name__ == '__main__':
    main()
    
