#!/usr/bin/env python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from statistics import mean
from sklearn import preprocessing
import tensorflow_probability as tfp
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor

def normalize(x):
    names = x.columns
    scaler = preprocessing.StandardScaler().fit(x)
    x = pd.DataFrame(scaler.transform(x))
    x.columns = names
    return x

@tf.function
def corr_corf(y_true, y_pred):
    if tfp.stats.variance(y_pred) == 0:
        return 0.0
    else:
        return tfp.stats.correlation(y_true, y_pred)

def pre_defined(args):
    global important_genes

    with open('important_genes.pkl', 'rb') as f:
        important_genes = pickle.load(f)

    all_models = os.listdir('NNmodels')

    if args.new_circ:
        new_circ = pd.read_csv(args.new_circ)
        new_circ = normalize(new_circ)

    with Pool(processes = args.num_threads) as pool:
        results = [pool.apply_async(predict_with_pre_defined, (model,)) \
            for model in all_models]
        pred = []
        for result in results: 
            pred.append(result.get())
    pred = pd.DataFrame(pred)
    pred.index = all_models
    pred = pred.T
    if args.new_circ:
        cc = pred.corrwith(new_circ)
        cc.to_csv(args.output_prefix + '_cc.csv', header = ['CC'])
    if not args.disable_value_output or not args.new_circ:
        pred.to_csv(args.output_prefix + '_value.csv')

def predict_with_pre_defined(m):
    from tensorflow.keras import models, callbacks, utils
    import tensorflow_addons as tfa
    utils.get_custom_objects()['gelu'] = tfa.activations.gelu
    tfa.register_all()

    model = models.load_model(os.path.join('NNmodels', m), \
        custom_objects = {"corr_corf" : corr_corf})
    return model.predict(np.array(new_pcg.loc[:, important_genes[m]])).flatten()

def custom(args):
    global train_circ, train_pcg

    train_circ = pd.read_csv(args.train_circ)
    train_circ = normalize(train_circ)
    train_pcg = pd.read_csv(args.train_pcg)
    train_pcg = normalize(train_pcg)

    if args.new_circ:
        new_circ = pd.read_csv(args.new_circ)
        new_circ = normalize(new_circ)

    with Pool(processes = args.num_threads) as pool:
        results = [pool.apply_async(predict_with_custom, (column,)) \
            for column in train_circ.columns]
        pred = []
        for result in results: 
            pred.append(result.get())
    pred = pd.DataFrame(pred)
    pred.index = train_circ.columns
    pred = pred.T
    if args.new_circ:
        cc = pred.corrwith(new_circ)
        cc.to_csv(args.output_prefix + '_cc.csv', header = ['CC'])
    if not args.disable_value_output or not args.new_circ:
        pred.to_csv(args.output_prefix + '_value.csv')
    
def predict_with_custom(column):
    from tensorflow.keras import models, callbacks, utils
    import tensorflow_addons as tfa
    utils.get_custom_objects()['gelu'] = tfa.activations.gelu

    y = train_circ[column]
    regr = RandomForestRegressor(n_jobs = -1)
    regr.fit(train_pcg, y)
    x_corf = train_pcg.iloc[:, regr.feature_importances_.argsort()[::-1][:100]]

    model = models.load_model('final_1.h5', \
        custom_objects = {"corr_corf" : corr_corf})
    try:
        model.fit(np.array(x_corf), y.values, epochs=1000, \
            validation_split = 0.2, batch_size = 32, verbose = 0, \
                callbacks = callbacks.EarlyStopping('val_corr_corf', \
                    mode = 'max', restore_best_weights = True, patience = 10, \
                        baseline = 0, min_delta=0))
    except TypeError:
        pass
    
    return model.predict(np.array(new_pcg.iloc[:, \
        regr.feature_importances_.argsort()[::-1][:100]])).flatten()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-dataset Validation by ICE.')
    subparsers = parser.add_subparsers(title='subcommands', help='mode of cross-dataset validation')

    parser_pre_defined = subparsers.add_parser('pre_defined', help = 'validate by predicted value of pre-defined (pan-cancer) weight')
    parser_pre_defined.add_argument('new_pcg', help = 'new Protein Coding Gene expression matrix for prediting')
    parser_pre_defined.add_argument('--new_circ', help = 'measured value matrix of circRNAs')
    parser_pre_defined.add_argument('--output_prefix', default = 'output', help = 'output file prefix. Default is "output"')
    parser_pre_defined.add_argument('--num_threads', default = 4, type = int, help = 'cpu number for parallel processing. Default is 4')
    parser_pre_defined.add_argument('--disable_value_output', dest = "disable_value_output", action='store_true', \
        help = 'cancel output of predicted value. It is invalid when `--new_circ` does not exist')
    parser_pre_defined.set_defaults(func = pre_defined)

    parser_custom = subparsers.add_parser('custom', help = 'validate by predicted value with re-fitted models from custom PCG matrix')
    parser_custom.add_argument('train_pcg', help = 'existed Protein Coding Gene expression matrix for fitting')
    parser_custom.add_argument('train_circ', help = 'existed circRNA expression matrix for fitting')
    parser_custom.add_argument('new_pcg', help = 'new Protein Coding Gene expression matrix for prediting')
    parser_custom.add_argument('--new_circ', help = 'measured value matrix of circRNAs')
    parser_custom.add_argument('--output_prefix', default = 'output', help = 'output file prefix. Default is "output"')
    parser_custom.add_argument('--num_threads', default = 4, type = int, help = 'cpu number for parallel processing. Default is 4')
    parser_custom.add_argument('--disable_value_output', dest = "disable_value_output", action='store_true', \
        help = 'cancel output of predicted value. It is invalid when `--new_circ` does not exist')
    parser_custom.set_defaults(func = custom)

    parser.set_defaults(disable_value_output=False)
    args = parser.parse_args()

    new_pcg = pd.read_csv(args.new_pcg)
    new_pcg = normalize(new_pcg)

    args.func(args)
