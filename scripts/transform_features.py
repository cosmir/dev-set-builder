#!/usr/bin/env python
'''Convert the AudioSet features into an sklearn-friendly format.

Example
-------
$ ./transform_features.py /path/to/audioset_v1_embeddings/unbal_train \
    /path/to/outputs --prefix audioset_train_

Will produce two files:
    /path/to/outputs/audioset_train_features.npy (~2.4GB)
    /path/to/outputs/audioset_train_labels.csv (~80MB)
'''

import argparse
import glob
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

START = 'start_time_seconds'
VID_ID = 'video_id'
LABEL = 'labels'
FEAT_NAME = 'audio_embedding'
TIME = 'time'


def filebase(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def safe_makedirs(dpath):
    if not os.path.exists(dpath) and dpath:
        os.makedirs(dpath)


def bytestring_to_record(example):
    """Convert a serialized tf.SequenceExample to a dict of python/numpy types.

    Parameters
    ----------
    example : str
        Serialized tf.SequenceExample

    feature_shape : tuple, len=2
        Expected shape of the feature array.

    on_mismatch : str, default='skip'
        Behavior of the function in the event that the features are *not* the
        expected shape, one of:

         * 'skip': Return an empty object
         * 'coerce': Backfill with zeros or slice, depending on the mismatch;
            In the event that the observed features are smaller, will use
            `fill_value` to backfill.
         * 'strict': Will raise a ValueError if there is any discrepancy.

    fill_value : int, default=128
        Fill-value for feature arrays that need to be extended. Note that in
        the AudioSet representation, features are uint8 encoded, i.e. 128
        corresponds to zero-mean.

    Returns
    -------
    features : np.array, shape=(n, 128)
        Array of feature coefficients over time (axis=0).

    meta : pd.DataFrame, len=n
        Corresponding labels and metadata for these features.
    """
    rec = tf.train.SequenceExample.FromString(example)
    start_time = rec.context.feature[START].float_list.value[0]
    video_id = rec.context.feature[VID_ID].bytes_list.value[0].decode('utf-8')
    labels = list(rec.context.feature[LABEL].int64_list.value)
    features = [b.bytes_list.value
                for b in rec.feature_lists.feature_list[FEAT_NAME].feature]
    features = np.asarray([np.frombuffer(_[0], dtype=np.uint8)
                           for _ in features])
    if features.ndim == 1:
        features = features.reshape(1, -1)

    meta = pd.DataFrame.from_records(
        data=[{VID_ID: video_id, LABEL: labels,
               TIME: np.uint8(start_time + t)}
              for t in range(len(features))])
    return features, meta


def load_tfrecord(fname, n_jobs=1, verbose=0):
    """Transform a YouTube-8M style tfrecord file to numpy / pandas objects.

    Parameters
    ----------
    fname : str
        Filepath on disk to read.

    n_jobs : int, default=-2
        Number of cores to use, defaults to all but one.

    verbose : int, default=0
        Verbosity level for loading.

    Returns
    -------
    features : np.array, shape=(n_obs, n_coeffs)
        All observations, concatenated together,

    meta : pd.DataFrame
        Table of metadata aligned to the features, indexed by `filebase.idx`
    """
    dfx = delayed(bytestring_to_record)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    results = pool(dfx(x) for x in tf.python_io.tf_record_iterator(fname))
    features = np.concatenate([xy[0] for xy in results], axis=0)
    meta = pd.concat([xy[1] for xy in results], axis=0, ignore_index=True)
    return features, meta


def convert_dataset(filenames, outdir, prefix='', n_jobs=-2, verbose=1):
    """Convert the TF version of AudioSet to NumPy / Pandas formats.

    Parameters
    ----------
    filenames : iterable of str
        Collection of filepaths to convert.

    outdir : str
        Root directory at which to write outputs.

    prefix : str, default=''
        Optional string with which to prefix created files, like:
            {prefix}features.npy, {prefix}labels.csv

    feature_shape, on_mismatch, fill_value
        See `load_tfrecord`

    n_jobs : int, default=-2
        Number of parallel jobs to run.

    verbose : int, default=1
        Verbosity level, see joblib.Parallel for more info.

    Returns
    -------
    success : bool
        True if both files are written successfully.
    """
    dfx = delayed(load_tfrecord)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    kwargs = dict(n_jobs=1, verbose=0)
    results = pool(dfx(fn, **kwargs) for fn in filenames)

    features = np.concatenate([xy[0] for xy in results], axis=0)
    safe_makedirs(outdir)
    features_file = os.path.join(outdir, "{}features.npy".format(prefix))
    np.save(features_file, features)
    meta = pd.concat([xy[1] for xy in results], axis=0, ignore_index=True)
    meta_file = os.path.join(outdir, "{}labels.csv".format(prefix))
    meta.to_csv(meta_file, index=False)
    return all([os.path.exists(fn) for fn in (features_file, meta_file)])


def process_args(args):

    parser = argparse.ArgumentParser(
        description='Transform the AudioSet features into a more '
                    'sklearn-friendly format.')

    parser.add_argument(dest='feature_path', action='store',
                        type=str, help='Path to a directory of `*.tfrecords`')

    parser.add_argument(dest='output_path', type=str, action='store',
                        help='Path to store output npy and csv files')

    parser.add_argument('--prefix', type=str, default='',
                        help='File prefix for writing outputs.')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    tf_files = glob.glob(os.path.join(args.feature_path, "*.tfrecord"))
    convert_dataset(tf_files, args.output_path, args.prefix)
