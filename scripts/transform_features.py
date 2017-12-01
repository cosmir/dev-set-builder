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
import audioset.util as util
import glob
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
import sys


def main(filenames, outdir, prefix='', n_jobs=-2, verbose=1):
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
    dfx = delayed(util.load_tfrecord)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    kwargs = dict(n_jobs=1, verbose=0)
    results = pool(dfx(fn, **kwargs) for fn in filenames)

    features = np.concatenate([xy[0] for xy in results], axis=0)
    util.safe_makedirs(outdir)
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
    main(tf_files, args.output_path, args.prefix)
