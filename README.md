# dev-set-builder
Boostrapping weak multi-instrument classifiers to build a development dataset with the open-mic taxonomy.


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