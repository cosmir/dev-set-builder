import pytest

import librosa.output
import numpy as np
import os
import pandas as pd

import featurefy
import filter_with_classes
import transform_features


def test_featurefy_main(audio_file, tmpdir):
    success = featurefy.main([audio_file], str(tmpdir))
    assert all(success)


def test_featurefy_main_garbage_audio(tmpdir):
    audio_file = os.path.join(str(tmpdir), "empty_file.wav")
    librosa.output.write_wav(audio_file, np.array([]), 44100)

    success = featurefy.main([audio_file], str(tmpdir))
    assert not all(success)


def test_transform_features_main(tfrecords, tmpdir):
    assert transform_features.main(tfrecords, str(tmpdir), prefix='dummy_')


def test_filter_with_classes_main(audioset_sample, openmic_video_labels,
                                  openmic_class_map, tmpdir):
    features = np.load(audioset_sample[0], mmap_mode='r')
    labels = pd.read_csv(audioset_sample[1])
    num_background = 10
    assert filter_with_classes.main(
        features, labels, openmic_video_labels, 'video_id',
        openmic_class_map['classes'], num_background, str(tmpdir),
        weak_null_classes=set([137]), prefix='openmic_', random_state=123)

    x_in = np.load(os.path.join(str(tmpdir), "openmic_features.npy"))
    y_true = np.load(os.path.join(str(tmpdir), "openmic_classes.npy"))
    assert len(x_in) == len(y_true) > 500
    assert y_true.sum() > 10
    assert (y_true.sum(axis=1) == 0).sum() >= num_background
