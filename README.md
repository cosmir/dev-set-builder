# dev-set-builder
Boostrapping weak multi-instrument classifiers to build a development dataset with the open-mic taxonomy.

[![Build Status](https://travis-ci.org/cosmir/dev-set-builder.svg?branch=master)](https://travis-ci.org/cosmir/dev-set-builder)
[![Coverage Status](https://coveralls.io/repos/github/cosmir/dev-set-builder/badge.svg?branch=master)](https://coveralls.io/github/cosmir/dev-set-builder?branch=master)

## Install

```bash
$ pip install -e .
```

## Run it!

```bash
$ echo "my/audio/file.wav,my/jams/file.jams" > filelist.csv
$ python scripts/featurefy.py filelist.csv ./outputs
$ ls ./outputs
corpus_file.h5
```

## MNIST-ifying the AudioSet

The original TensorFlow-friendly dump of the VGGish features for AudioSet is made [freely available online](https://research.google.com/audioset/download.html). Here, VGGish features are sharded into 4096 nested `tf.SequenceExample`s, with each shard containing several hundred excerpts (between 300-1000, averaging around 900).

While this works well for TensorFlow, the format can be a bit overwraught for more Pythonic implementations, e.g. `sklearn`. Here, we provide a `transforms_features.py` script that unpacks the features into a large(ish) NumPy tensor (2.4GB) with shape `[examples * time, feature]`, and a CSV file of metadata, tracking both labels and provenance info, with the columns `[index, labels, time, video_id]`. The index of this table corresponds exactly to the feature array.

These artifacts are made freely available here:

* [AudioSet Train Features]()
* [AudioSet Train Labels]()
* [AudioSet Test Features]()
* [AudioSet Test Labels]()

![]()  # Image of data snapshot goes here.

### OpenMIC-23 Subset

As a final preprocessing step to build our development set, the full AudioSet is filtered on OpenMIC instrument classes and the labels are transformed into a sparse indicator (boolean) array, for easy use with Pythonic ML frameworks, e.g. `sklearn`.

#### Training Distribution

The OpenMIC subset for the unbalanced training set looks like the following:

| Instrument           | Expected | Actual   | Missing |
| -------------------- | --------:| --------:| -------:|
| guitar               |    56926 |    56489 |   0.77% |
| violin               |    28065 |    28001 |   0.23% |
| drums                |    26331 |    26076 |   0.97% |
| piano                |    12744 |    12654 |   0.71% |
| bass                 |     8549 |     8428 |   1.42% |
| mallet_percussion    |     7257 |     7128 |   1.78% |
| voice                |     6611 |     6549 |   0.94% |
| cymbals              |     5365 |     5301 |   1.19% |
| ukulele              |     5232 |     5172 |   1.15% |
| cello                |     5215 |     5148 |   1.28% |
| synthesizer          |     4981 |     4921 |   1.20% |
| flute                |     4721 |     4659 |   1.31% |
| trumpet              |     3771 |     3707 |   1.70% |
| organ                |     3578 |     3458 |   3.35% |
| saxophone            |     3013 |     2950 |   2.09% |
| accordion            |     2833 |     2772 |   2.15% |
| trombone             |     2731 |     2666 |   2.38% |
| banjo                |     2396 |     2336 |   2.50% |
| mandolin             |     2312 |     2252 |   2.60% |
| harmonica            |     2156 |     2095 |   2.83% |
| clarinet             |     2061 |     1998 |   3.06% |
| harp                 |     1983 |     1921 |   3.13% |
| bagpipes             |     1715 |     1655 |   3.50% |
| none of the above    |  1841243 |     8000 |    --   |

Note that we backfill with 8000 randomly sampled observations, chosen to be reasonably disjoint with the given instrument classes.