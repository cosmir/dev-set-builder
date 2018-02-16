#!/usr/bin/env python
# coding: utf8
'''Obtain the most likely k instruments on a dataset
subset and group excerpts by instrument similarity
'''

import argparse
import numpy as np
import pandas as pd
import sys


def params(args):

    parser = argparse.ArgumentParser(description='Obtain the most likely instruments on a dataset subset and group excerpts by instrument similarity')
    parser.add_argument('name_subset', type=str, help='Path to IRMAS data')
    parser.add_argument('name_maxagg', type=str, help='Path to the full dataset')
    parser.add_argument('name_output', type=str, help='Path to the output subset')

    return vars(parser.parse_args(args))


def get_max_inst(name_subset, name_maxagg, name_output):

    k = 7  # to do: allow to change this
    df = pd.read_csv(name_maxagg, index_col=0)
    idx = pd.read_json(name_subset, typ='Series')
    df_subset = df.loc[idx]
    ind_sub = df_subset.index
    inst_max = np.empty([len(ind_sub), k], dtype=object)
    for repeat in range(k):
        idm = df_subset.T.idxmax()
        inst_max[:, repeat] = np.copy(idm)
        for count in range(len(idm)):
            df_subset.at[ind_sub[count], idm[count]] = -1

    df_inst_max = pd.DataFrame(inst_max, columns=list('1234567'), index=idx)
    df_inst_max_sort = df_inst_max.sort_values(by=['1', '2', '3', '4', '5', '6', '7'])
    ind_sort = df_inst_max_sort.index
    count = 0
    group_nb = np.zeros(len(df_inst_max_sort))
    for ii in range(len(df_inst_max_sort) - 1):
        group_nb[ii] = count
        if ~(df_inst_max_sort.loc[ind_sort[ii]] == df_inst_max_sort.loc[ind_sort[ii + 1]]).all():
            count += 1

    df_inst_max_sort['group'] = pd.Series(group_nb, index=df_inst_max_sort.index)
    df_inst_max_sort.to_pickle(name_output)


if __name__ == '__main__':
    args = params(sys.argv[1:])
    get_max_inst(args['name_subset'], args['name_maxagg'], args['name_output'])
