import pytest

import glob
import json
import os


@pytest.fixture()
def data_dir():
    return os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')


@pytest.fixture()
def audio_file(data_dir):
    return os.path.join(data_dir, '6457__dobroide__sunday-02.mp3')


@pytest.fixture()
def tfrecords(data_dir):
    return glob.glob(os.path.join(data_dir, "tfrecords/*.tfrecord"))


@pytest.fixture()
def openmic_class_map(data_dir):
    return json.load(open(os.path.join(data_dir, "openmic25_class_map.json")))


@pytest.fixture()
def openmic_index(data_dir):
    return json.load(open(os.path.join(data_dir, "openmic25_index.json")))


@pytest.fixture()
def openmic_video_labels(data_dir):
    return json.load(open(os.path.join(data_dir,
                                       "openmic25_video_labels.json")))


@pytest.fixture()
def audioset_sample(data_dir):
    return (os.path.join(data_dir, "audioset_sample/dummy_features.npy"),
            os.path.join(data_dir, "audioset_sample/dummy_labels.csv"))
