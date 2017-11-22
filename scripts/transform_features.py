#!/usr/bin/env python
'''Convert the AudioSet features into an sklearn-friendly format.

Example
-------
$ ./transform_features.py /path/to/audioset_v1_embeddings/unbal_train \
    /path/to/outputs

Will produce two files:
    /path/to/outputs/audioset_features_train.npy (~2.4GB)
    /path/to/outputs/audioset_labels_train.csv (~80MB)
'''

import argparse
import glob
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import warnings

START = 'start_time_seconds'
VID_ID = 'video_id'
LABEL = 'labels'
FEAT_NAME = 'audio_embedding'


def filebase(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def safe_makedirs(dpath):
    if not os.path.exists(dpath) and dpath:
        os.makedirs(dpath)


def bytestring_to_record(example, feature_shape=(10, 128), on_mismatch='skip',
                         fill_value=128):
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
    data : dict
        Object containing the named objects of the tf.SequenceExample. May be
        empty if the features are mismatched.
    """
    rec = tf.train.SequenceExample.FromString(example)
    start_time = rec.context.feature[START].float_list.value[0]
    video_id = rec.context.feature[VID_ID].bytes_list.value[0].decode('utf-8')
    labels = list(rec.context.feature[LABEL].int64_list.value)
    features = [b.bytes_list.value
                for b in rec.feature_lists.feature_list[FEAT_NAME].feature]
    X = np.asarray([np.frombuffer(_[0], dtype=np.uint8) for _ in features])

    data = dict()
    if X.shape == feature_shape:
        data.update(**{FEAT_NAME: X, VID_ID: video_id,
                       LABEL: labels, START: start_time})
    elif on_mismatch == 'strict':
        raise ValueError("Expected features to have shape {}; actual {}"
                         .format(feature_shape, X.shape))
    elif on_mismatch == 'skip':
        warnings.warn("Expected features to have shape {}; actual {}"
                      .format(feature_shape, X.shape))

    elif on_mismatch == 'coerce':
        raise NotImplementedError("`coerce` not yet implemented :o(")

    return data


def load_tfrecord(fname, feature_shape=(10, 128), on_mismatch='skip',
                  fill_value=128, n_jobs=1, verbose=0):
    """Transform a YouTube-8M style tfrecord file to numpy / pandas objects.

    Parameters
    ----------
    fname : str
        Filepath on disk to read.

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

    n_jobs : int, default=-2
        Number of cores to use, defaults to all but one.

    verbose : int, default=0
        Verbosity level for loading.

    Returns
    -------
    features : np.array, shape=(n_samples, n_steps, n_coeffs)
        Batch of observations.

    meta : pd.DataFrame
        Table of metadata aligned to the features, indexed by `filebase.idx`
    """
    dfx = delayed(bytestring_to_record)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    kwargs = dict(feature_shape=feature_shape, fill_value=fill_value,
                  on_mismatch=on_mismatch)
    records = pool(dfx(x, **kwargs)
                   for x in tf.python_io.tf_record_iterator(fname))

    # Unpack features and skip any malformed objects.
    features = np.array([data.pop(FEAT_NAME) for data in records if data])
    key = filebase(fname)
    index = ["{}.{:4d}".format(key, n)
             for n, data in enumerate(records) if data]
    meta = pd.DataFrame.from_records(filter(None, records), index=index)
    return features, meta


def convert_dataset(filenames, outdir, feature_shape=(10, 128),
                    on_mismatch='skip', fill_value=128, n_jobs=-2, verbose=1):
    dfx = delayed(load_tfrecord)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    kwargs = dict(feature_shape=feature_shape, fill_value=fill_value,
                  on_mismatch=on_mismatch, n_jobs=1, verbose=0)
    results = pool(dfx(fn, **kwargs) for fn in filenames)

    X = np.concatenate([xy[0] for xy in results], axis=0)
    safe_makedirs(outdir)
    x_out = os.path.join(outdir, "audioset_features_train.npy")
    np.save(x_out, X)
    Y = pd.concat([xy[1] for xy in results])
    y_out = os.path.join(outdir, "audioset_labels_train.csv")
    Y.to_csv(y_out, index_label="index")


def process_args(args):

    parser = argparse.ArgumentParser(
        description='Transform the AudioSet features into a more '
                    'sklearn-friendly format.')

    parser.add_argument(dest='feature_path', action='store',
                        type=str, help='Path to a directory of tfrecords')

    parser.add_argument(dest='output_path', type=str, action='store',
                        help='Path to store output npy and csv files')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    tf_files = glob.glob(os.path.join(args.feature_path, "*.tfrecord"))
    convert_dataset(tf_files, args.output_path)
