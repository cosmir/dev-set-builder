import pytest

import glob
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
