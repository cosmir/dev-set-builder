import pytest

import json
import numpy as np
import os
import yaml

import audioset.openmic


@pytest.fixture()
def model_kwargs():
    return yaml.load("""folds:
  train: [0, 1, 2]
  valid: [3]
  test: [4]
model_args:
  n_in: 30
  width: [1024, 1024, 17]
  dropout: [0.5, 0.5, 0.5]
  activation: ['relu', 'relu', 'sigmoid']
  batch_norm: [True, True, False]
  opt_kwargs:
    name: "Adam"
    lr: 0.0001
    beta_1: 0.5
fit_args:
  epochs: 3
  batch_size: 64
  class_weighted: True
outputs:
  name: "deleteme"
  dirname: "./"
  checkpoint_fmt: "weights-{}.h5"
  """)


def test_build_model():
    assert audioset.openmic.build_model(128, [23], [0.5], ['linear'], [True],
                                        dict(name='SGD'))
    with pytest.raises(ValueError):
        audioset.openmic.build_model(128, [10, 23], [0.5], ['linear'], [True],
                                     dict(name='SGD'))


def test_fit_model(model_kwargs, tmpdir):

    k_dim = 30
    num_samples = 500
    num_classes = 10
    X = np.random.uniform(size=(num_samples, k_dim))
    Y = np.zeros([num_samples, 17])
    Y[np.arange(num_samples),
      np.random.randint(0, num_classes, size=(num_samples,))] = 1
    folds = np.random.randint(0, 5, size=(num_samples,))

    model_kwargs['outputs']['dirname'] = os.path.join(str(tmpdir))
    os.makedirs(os.path.join(model_kwargs['outputs']['dirname'],
                             model_kwargs['outputs']['name']))
    outputs = audioset.openmic.fit_model(X, Y, folds, model_kwargs,
                                         list(range(num_classes)))
    assert len(outputs) == 8  # epochs + 5


def test_compute_stats():
    y_true = np.array([[1, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 1]])
    y_proba = np.array([[0.8, 0.2, 0.1, 0.0],
                        [0.0, 0.2, 0.9, 0.9],
                        [0.1, 0.7, 0.1, 0.2],
                        [0.1, 0.0, 0.2, 0.5]])
    folds = {'all': np.array([True, True, True, True])}
    labels = list('abcd')
    thresholds = [0.5, [0.4, 0.5, 0.6, 0.2]]
    exp_micro_f1s = [0.5, 0.66]
    exp_micro_aucs = [0.85, 0.85]
    for t, f1, auc in zip(thresholds, exp_micro_f1s, exp_micro_aucs):
        df = audioset.openmic.compute_stats(y_true, y_proba, folds, labels, t)
        # 4 PRF measures, 3 ROC measures, and `k` classwise scores.
        assert len(df) == 7 + y_true.shape[1]
        assert df.loc['prf_micro_all'].f1 >= f1
        assert df.loc['auc_micro_all'].roc_auc_score >= auc

    with pytest.raises(ValueError):
        audioset.openmic.compute_stats(y_true, y_proba, folds, labels,
                                       [0.5, 0.2])


def test_generate_configs():
    num_configs = 5
    k_folds = 4
    configs = audioset.openmic.generate_configs(num_configs, 23, 'deleteme-',
                                                random_state=13579)
    assert len(configs) == num_configs * k_folds
    assert len(configs) == len(set([kw['outputs']['name'] for kw in configs]))

    for kwargs in configs[::k_folds]:
        assert audioset.openmic.build_model(**kwargs['model_args']) is not None
