import pytest

import featurefy
import transform_features


def test_featurefy_main(audio_file, tmpdir):
    success = featurefy.main([audio_file], str(tmpdir))
    assert all(success)


def test_transform_features_main(tfrecords, tmpdir):
    assert transform_features.main(tfrecords, str(tmpdir), prefix='dummy_')
