import pytest

import featurefy


def test_run_model(audio_file, tmpdir):
    success = featurefy.run_model([audio_file], str(tmpdir))
    assert all(success)
