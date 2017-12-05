#!/usr/bin/env python
'''Compute VGGish features for a batch of files

Example:
$ cd {repo_root}
$ ./scripts/featurefy.py input_files.csv ./output_dir

Note that the CSV file should be headerless, and contain two columns:
    - Path to an audio file
    - Path to a jams file

Each jams file must contain at least one annotation in the `tag_openmic25`
namespace.
'''

import argparse

import os
import sys

import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import audioset
import audioset.vggish_input
import audioset.vggish_slim
import audioset.vggish_postprocess


def load_input(filename):

    y, sr = librosa.load(filename, sr=audioset.SAMPLE_RATE, mono=True)
    y = librosa.util.normalize(y)

    return audioset.vggish_input.waveform_to_examples(y, sr)


def main(files_in, outpath):

    pproc = audioset.vggish_postprocess.Postprocessor(audioset.PCA_PARAMS)
    success = []
    with tf.Graph().as_default(), tf.Session() as sess:

        audioset.vggish_slim.define_vggish_slim(training=False)
        audioset.vggish_slim.load_vggish_slim_checkpoint(
            sess, audioset.MODEL_PARAMS)
        features_tensor = sess.graph.get_tensor_by_name(
            audioset.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            audioset.OUTPUT_TENSOR_NAME)

        for file_in in tqdm(files_in):

            file_out = os.path.join(
                outpath,
                os.path.extsep.join([os.path.basename(file_in), 'npz']))
            input_data = load_input(file_in)

            [embedding] = sess.run([embedding_tensor],
                                   feed_dict={features_tensor: input_data})

            emb_pca = pproc.postprocess(embedding)

            np.savez(file_out, features=embedding, features_z=emb_pca)
            success.append(os.path.exists(file_out))
    return success


def process_args(args):

    parser = argparse.ArgumentParser(description='VGGish feature extractor')

    parser.add_argument(dest='input_list', action='store',
                        type=str, help='Path to input CSV file')

    parser.add_argument(dest='output_path', type=str, action='store',
                        help='Path to store output files in NPZ format')

    return parser.parse_args(args)


def load_files_in(input_list):

    files_in = pd.read_table(input_list, header=None)
    return list(files_in[0])


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    files_in = load_files_in(args.input_list)

    main(files_in, args.output_path)
