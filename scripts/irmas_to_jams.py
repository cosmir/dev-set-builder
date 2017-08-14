#!/usr/bin/env python

import argparse
import sys
import os

from tqdm import tqdm
from joblib import Parallel, delayed

os.environ.setdefault('JAMS_SCHEMA_DIR', '../schema')

import jams
import librosa


# Filename encodings of instruments => actual words
INSTRUMENT_MAP = {'cel': 'cello',
                  'cla': 'clarinet',
                  'dru': 'drums',
                  'flu': 'flute',
                  'gac': 'guitar (acoustic)',
                  'gel': 'guitar (electric)',
                  'org': 'organ',
                  'pia': 'piano',
                  'sax': 'saxophone',
                  'tru': 'trumpet',
                  'vio': 'violin',
                  'voi': 'voice'}

BASE_VOCAB = sorted(INSTRUMENT_MAP.keys())


def params(args):

    parser = argparse.ArgumentParser(description='Convert IRMAS to JAMS format')
    parser.add_argument('path', type=str, help='Path to IRMAS data')
    parser.add_argument('-j', '--num-jobs', dest='num_jobs',
                        default=1, type=int, help='Number of parallel jobs')

    return vars(parser.parse_args(args))


def jamify_train(infile):

    dirname = os.path.dirname(infile)
    label = os.path.basename(dirname)

    basename = os.path.basename(infile)
    root = os.path.splitext(basename)[0]
    jam_out = os.path.join(dirname, os.path.extsep.join([root, 'jams']))

    duration = librosa.get_duration(filename=infile)

    J = jams.JAMS()
    J.file_metadata.duration = duration
    J.file_metadata.title = root

    ann = jams.Annotation(namespace='tag_irmas_instruments',
                          duration=duration)
    ann.append(time=0, duration=duration, value=INSTRUMENT_MAP[label], confidence=None)

    if '[dru]' in basename:
        ann.append(time=0, duration=duration, value='drums', confidence=None)

    ann.annotation_metadata.corpus = 'IRMAS'
    ann.annotation_metadata.data_source = 'training'
    J.annotations.append(ann)
    J.save(jam_out)


def process_train(path, num_jobs):

    all_files = jams.util.find_with_extension(os.path.join(path,
                                                           'IRMAS-TrainingData'),
                                              'wav', depth=2)

    Parallel(n_jobs=num_jobs)(delayed(jamify_train)(infile) for infile in tqdm(all_files))


if __name__ == '__main__':
    args = params(sys.argv[1:])
    process_train(args['path'], args['num_jobs'])
