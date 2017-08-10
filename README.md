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