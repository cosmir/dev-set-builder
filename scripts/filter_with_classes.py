#!/usr/bin/env python
'''Convert the AudioSet features into an sklearn-friendly format.

Example
-------
$ ./openmic_subset.py /path/to/audioset_features.npy \
    /path/to/audioset_labels.csv \
    openmic25_index.json \
    /path/to/outputs --prefix openmic_

Will produce two files:
    /path/to/outputs/openmic_features.npy
    /path/to/outputs/openmic_labels.npy
'''

import argparse
import json
import numpy as np
import os
import pandas as pd
import sys


def main(src_features, src_labels, subset_index, column,
         class_map, outdir, prefix=''):
    """Produce a filtered subset given a dataset and a set of IDs.

    Parameters
    ----------
    src_features : np.ndarray, shape=(n, d)
        The feature array.

    src_labels : pd.DataFrame, len=n
        Corresponding metadata, aligned to the source feature array.

    subset_index : dict
        Map of generic IDs to class labels. The gid namespace should match that
        specified by `column`.

    column : str
        Column to use for filtering gids from the source label dataframe.

    class_map : dict
        Mapping between labels (in subset_index) to integer positions.

    outdir : str
        Path for writing the various outputs.

    prefix : str, default=''
        Optional string with which to prefix created files, like:
            {prefix}features.npy, {prefix}labels.csv, {prefix}classes.npy

    Returns
    -------
    success : bool
        True if all files were created correctly.
    """

    gids = sorted(list(subset_index.keys()))

    dst_labels = pd.DataFrame(data=dict(keep=True), index=gids)
    dst_labels = src_labels.join(dst_labels, on=column, how='inner')
    del dst_labels['keep']

    dst_features = src_features[dst_labels.index.values]
    features_file = os.path.join(outdir, "{}features.npy".format(prefix))
    np.save(features_file, dst_features)

    labels_file = os.path.join(outdir, "{}labels.csv".format(prefix))
    dst_labels.to_csv(labels_file, index=True)

    y_true = np.zeros([len(dst_features), len(class_map)], dtype=bool)
    for n, gid in enumerate(dst_labels[column]):
        for y_label in subset_index[gid]:
            y_true[n, class_map[y_label]] = True

    y_true_file = os.path.join(outdir, "{}classes.npy".format(prefix))
    np.save(y_true_file, y_true)

    output_files = (features_file, labels_file, y_true_file)
    return all([os.path.exists(fn) for fn in output_files])


def process_args(args):

    parser = argparse.ArgumentParser(
        description='Transform the src features into a more '
                    'sklearn-friendly format.')

    parser.add_argument(dest='src_feature_file', action='store',
                        type=str, help='Path to a NPY file of features.')
    parser.add_argument(dest='src_labels_file', action='store',
                        type=str, help='Path to a CSV file of metadata.')
    parser.add_argument(dest='subset_file', type=str, action='store',
                        help='Path to a JSON file mapping gids '
                             'to class labels.')
    parser.add_argument(dest='column', type=str, action='store',
                        help='Column in the source table to filter on.')
    parser.add_argument(dest='class_map_file', type=str, action='store',
                        help='Path to a mapping of class labels to integers.')
    parser.add_argument(dest='output_path', type=str, action='store',
                        help='Path to store output npy and csv files')
    parser.add_argument('--prefix', type=str, default='',
                        help='File prefix for writing outputs.')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    src_features = np.load(args.src_feature_file, mmap_mode='r')
    src_labels = pd.read_csv(args.src_labels_file)
    with open(args.subset_file, 'r') as fp:
        subset_index = json.load(fp)

    with open(args.class_map_file, 'r') as fp:
        class_map = json.load(fp)

    main(src_features, src_labels, subset_index, args.column,
         class_map, args.outdir, args.prefix)
