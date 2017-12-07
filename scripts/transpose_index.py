#!/usr/bin/env python
# coding: utf8
'''Transpose a label-major index to video_id-major.

Example
-------
$ ./scripts/transpose_index.py \
    data/audioset/openmic25_label_index.json \
    data/audioset/openmic25_video_index.json
'''

import argparse
import json
import os
import sys


def main(label_index, video_index):
    openmic_idx = json.load(open(label_index))

    gid_map = dict()

    for label, instances in openmic_idx.items():
        for gid, t0, t1 in instances:
            if gid not in gid_map:
                gid_map[gid] = []
            gid_map[gid].append(label)

    with open(video_index, 'w') as fp:
        json.dump(gid_map, fp)

    return os.path.exists(video_index)


def process_args(args):

    parser = argparse.ArgumentParser(
        description='Transpose a label-major index to video_id-major.')

    parser.add_argument(dest='label_index', action='store',
                        type=str, help='Path to a label-major JSON index.')
    parser.add_argument(dest='video_index', action='store',
                        type=str, help='Path to the video-major JSON output.')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    success = main(args.label_index, args.video_index)
    sys.exit(0 if success else 1)
