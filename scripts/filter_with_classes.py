#!/usr/bin/env python
# coding: utf8
'''Reduce a dataset on a subset index with binary-encoded classes.

Example
-------
$ ./filter_with_classes.py /path/to/audioset_features.npy \
    /path/to/audioset_labels.csv \
    data/openmic25_video_labels.json \
    video_id \
    data/openmic25_class_map.json \
    /path/to/outputs \
    --prefix openmic_

Will produce three files:
    /path/to/outputs/openmic_features.npy - X_in, shape=[n, 128]
    /path/to/outputs/openmic_classes.npy - y_true, shape=[n, 23]
    /path/to/outputs/openmic_labels.csv - provenance metadata
'''

import argparse
import json
import numpy as np
import os
import pandas as pd
import sys


def parse_labels(s):
    return [int(v.strip("L")) for v in s.strip("[]").split(",")]


def main(src_features, src_labels, subset_index, column,
         class_map, num_background, outdir, weak_null_classes=None,
         prefix='', random_state=None):
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

    num_background : int
        Number of background class samples to draw.

    outdir : str
        Path for writing the various outputs.

    weak_null_classes : list or set, default=None
        Object containing integer label indices to be filtered when buidling.

    prefix : str, default=''
        Optional string with which to prefix created files, like:
            {prefix}features.npy, {prefix}labels.csv, {prefix}classes.npy

    random_state : int, default=None
        Seed to use for the random number generator.

    Returns
    -------
    success : bool
        True if all files were created correctly.
    """

    def is_strong_null(row):
        """Return True if the labels correspond to a strong null condition."""
        return not any([y in weak_null_classes
                        for y in parse_labels(row['labels'])])

    gids = sorted(list(subset_index.keys()))

    dst_labels = pd.DataFrame(data=dict(keep=True), index=gids)
    dst_labels = src_labels.join(dst_labels, on=column, how='inner')
    del dst_labels['keep']

    dst_index = dst_labels.index
    dst_features = src_features[dst_index]

    if num_background > 0:
        null_index = src_labels.index.difference(dst_index)

        # Need to keep labels..
        null_labels = src_labels.loc[null_index]
        del null_labels['time']
        null_labels.drop_duplicates(inplace=True)

        # Tag videos that are sufficiently strong nulls
        strong_null_index = null_labels.apply(is_strong_null, axis=1)
        strong_null_gids = null_labels[strong_null_index][column].values

        # Slice a subset of corresponding GIDs
        rng = np.random.RandomState(random_state)
        rng.shuffle(strong_null_gids)
        strong_null_labels = pd.DataFrame(
            data=dict(keep=True), index=strong_null_gids[:num_background])
        strong_null_labels = src_labels.join(strong_null_labels,
                                             on=column, how='inner')
        del strong_null_labels['keep']

        # Concatenate strong nulls
        dst_features = np.concatenate(
            [dst_features, src_features[strong_null_labels.index]], axis=0)
        dst_labels = pd.concat([dst_labels, strong_null_labels], axis=0)

    features_file = os.path.join(outdir, "{}features.npy".format(prefix))
    np.save(features_file, dst_features)

    labels_file = os.path.join(outdir, "{}labels.csv".format(prefix))
    dst_labels.to_csv(labels_file, index=True)

    y_true = np.zeros([len(dst_features), len(class_map)], dtype=bool)
    for n, gid in enumerate(dst_labels[column]):
        for y_label in subset_index.get(gid, []):
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
    parser.add_argument('--num_background', type=int, default=0,
                        help='Number of background samples to draw.')
    parser.add_argument('--weak_null_classes', type=str, default=None,
                        help='JSON object of index values to filter when '
                             'sampling a null set.')
    parser.add_argument('--prefix', type=str, default='',
                        help='File prefix for writing outputs.')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Seed for random subsample.')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    src_features = np.load(args.src_feature_file, mmap_mode='r')
    src_labels = pd.read_csv(args.src_labels_file)
    with open(args.subset_file, 'r') as fp:
        subset_index = json.load(fp)

    with open(args.class_map_file, 'r') as fp:
        class_map = json.load(fp)['classes']

    weak_null_classes = []
    if args.weak_null_classes:
        with open(args.weak_null_classes, 'r') as fp:
            weak_null_classes += json.load(fp)['index']

    main(src_features, src_labels, subset_index, args.column,
         class_map, args.num_background,
         weak_null_classes=weak_null_classes,
         outdir=args.outdir, prefix=args.prefix)
