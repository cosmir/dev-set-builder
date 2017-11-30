import pytest

import numpy as np
import os
import pandas as pd

import featurefy
import filter_with_classes
import transform_features


def test_featurefy_main(audio_file, tmpdir):
    success = featurefy.main([audio_file], str(tmpdir))
    assert all(success)


def test_transform_features_main(tfrecords, tmpdir):
    assert transform_features.main(tfrecords, str(tmpdir), prefix='dummy_')


def test_filter_with_classes_main(audioset_sample, openmic_video_labels,
                                  openmic_class_map, tmpdir):
    features = np.load(audioset_sample[0], mmap_mode='r')
    labels = pd.read_csv(audioset_sample[1])
    assert filter_with_classes.main(
        features, labels, openmic_video_labels, 'video_id',
        openmic_class_map, str(tmpdir), 'openmic_')

    x_in = np.load(os.path.join(str(tmpdir), "openmic_features.npy"))
    y_true = np.load(os.path.join(str(tmpdir), "openmic_classes.npy"))
    assert len(x_in) == len(y_true) > 500
    assert y_true.sum() > 10
