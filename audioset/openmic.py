import glob
import json
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import os
import pandas as pd
from sklearn import metrics

OPTS = {
    "Adam": Adam,
    "SGD": SGD,
    "RMSprop": RMSprop
}


def build_model(n_in, width, dropout, activation, batch_norm, opt_kwargs):
    """Construct a keras dense network from the given parameters.

    Parameters
    ----------
    n_in : int
        Dimensionality of the input vector.

    widths : array_like of ints, len=n
        Dimensionality of each dense layer.

    dropout : array_like of floats, len=n
        Rate parameters for dropout; skipped if 0.

    activation : array_like of str, len=n
        Activation functions to use for each layer.

    batch_norm : array_like of bools, len=n
        Layerwise application of batch norm.

    opt_kwargs : dict
        Object containing parameters for the optimization algorithm.

    Returns
    -------
    model : keras.Model
        Compiled model for training.
    """
    if any([len(width) != len(_) for _ in (dropout, activation, batch_norm)]):
        raise ValueError("Input arguments must all have equal length.")

    x_in = Input(shape=(n_in,))
    z_out = x_in
    for k, d, a, b in zip(width, dropout, activation, batch_norm):
        drop = Dropout(d)
        layer = Dense(k, activation='linear')
        bnorm = BatchNormalization(axis=-1, momentum=0.9,
                                   epsilon=1e-5, scale=True)
        act = Activation(a)

        if d:
            z_out = drop(z_out)
        z_out = layer(z_out)
        if b:
            z_out = bnorm(z_out)
        z_out = act(z_out)

    model = Model(inputs=x_in, outputs=z_out)
    opt = OPTS[opt_kwargs.pop('name')](**opt_kwargs)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fit_model(X, y_true, folds, model_kwargs, labels):
    """Routine to train a classifier given data and model parameters.

    Parameters
    ----------
    X : np.ndarray, shape=(n, d)
        Input features for training, where d is the feature dimensionality.

    y_true : np.ndarray, shape=(n, k), in [0, 1]
        Class indicator matrix, where k is the number of classes.

    folds : np.ndarray, shape=(n,)
        Integer folds corresponding to each datapoint.

    labels : array_like of str
        String labels corresponding to the classes in y_true.

    model_kwargs : dict
        Object defining the model to train; see `build_model`.

    Returns
    -------
    filepaths : list, len=epochs + 5
        Collection of files successfully generated during execution, including
         * parameter checkpoints
         * training history stats
         * model kwargs
         * network graph JSON
         * evaluation stats
         * probability
    """
    fold_idx = {}
    for name, idx in model_kwargs['folds'].items():
        fold_idx[name] = np.zeros_like(folds).astype(bool)
        for i in idx:
            fold_idx[name] |= (folds == i)

    model = build_model(**model_kwargs['model_args'])

    output_dir = os.path.join(model_kwargs['outputs']['dirname'],
                              model_kwargs['outputs']['name'])

    model_kwargs_file = os.path.join(output_dir, 'model_kwargs.json')
    with open(model_kwargs_file, "w") as fp:
        json.dump(model_kwargs, fp)

    fpath = os.path.join(
        output_dir,
        model_kwargs['outputs']['checkpoint_fmt'].format("{epoch:04d}"))

    saver = ModelCheckpoint(
        fpath, monitor='val_loss', verbose=1,
        save_best_only=False, save_weights_only=True,
        mode='auto', period=1)

    kwargs = dict(batch_size=model_kwargs['fit_args']['batch_size'],
                  epochs=model_kwargs['fit_args']['epochs'])
    if model_kwargs['fit_args']['class_weighted']:
        counts = y_true[fold_idx['train']].sum(axis=0)
        counts /= counts.max()
        counts[counts == 0] = 1.0
        kwargs['class_weight'] = 1. / counts

    history = model.fit(
        X[fold_idx['train']], y_true[fold_idx['train']],
        validation_data=(X[fold_idx['valid']],
                         y_true[fold_idx['valid']]),
        callbacks=[saver], **kwargs)

    model_json = model.to_json()
    pred_file = os.path.join(output_dir, 'predictor.json')
    with open(pred_file, "w") as json_file:
        json_file.write(model_json)

    history_file = os.path.join(output_dir, 'fit_history.json')
    with open(history_file, 'w') as fp:
        json.dump(history.history, fp)

    # Keras writes 1-indexed files...
    best_epoch = np.array(history.history['val_acc']).argmax() + 1
    best_param_file = fpath.format(epoch=best_epoch)
    model.load_weights(best_param_file)
    y_proba = model.predict(X, batch_size=8192)

    probability_file = os.path.join(output_dir, 'probabilities.npz')
    np.savez(probability_file, y_proba=y_proba, labels=labels)
    stats = compute_stats(y_true, y_proba, fold_idx, labels)
    stats_file = os.path.join(output_dir, 'stats.json')
    stats.to_json(stats_file)
    param_files = glob.glob(os.path.join(
        output_dir, model_kwargs['outputs']['checkpoint_fmt'].format("*")))
    return param_files + [stats_file, probability_file,
                          history_file, pred_file, model_kwargs_file]


def compute_stats(Y_true, Y_proba, folds, labels, thresholds=0.5):
    """Compute metrics between true classes and predicted probabilities.

    Parameters
    ----------
    Y_true : np.ndarray, shape=(n, k)
        True labels as a binary indicator matrix.

    Y_proba : np.ndarray, shape=(n, k)
        Likelihoods that an observation (axis=0) takes a given class (axis=1).

    folds : dict of split: np.ndarray, shape=(n,), dtype=bool
        Dictionary mapping split names to boolean indicator vectors, used to
        partition results by split.

    labels : array_like
        Class labels corresponding to the columns of Y.

    thresholds : scalar or np.ndarray with len=k
        Bias point(s) to use for the probabilities in `Y_proba`. If given as a
        vector, the bias points will be applied per-class.

    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame of metrics across folds and averages.
    """
    data = []
    index = []
    num_classes = Y_proba.shape[-1]

    # If scalar, map to a vector.
    thresholds = np.asarray(thresholds)
    if thresholds.ndim == 0:
        thresholds = np.zeros([num_classes]) + thresholds
    elif len(thresholds) != num_classes:
        raise ValueError("If `thresholds` is an array, it must equal "
                         "`Y_true.shape[1]`")

    # Force to be 2d to make sure we broadcast correctly.
    if thresholds.ndim == 1:
        thresholds = thresholds.reshape(1, -1)

    Y_pred = (Y_proba > thresholds).astype(int)
    for fold, idx in folds.items():
        for ave in ['micro', 'macro', 'weighted']:
            scores = metrics.precision_recall_fscore_support(
                Y_true[idx], Y_pred[idx], average=ave)
            data.append(dict(average=ave, fold=fold, precision=scores[0],
                             recall=scores[1], f1=scores[2]))
            index += ["prf_{}_{}".format(ave, fold)]

        scores = metrics.precision_recall_fscore_support(
            Y_true[idx], Y_pred[idx])
        data.append(dict(fold=fold, precision=scores[0],
                         recall=scores[1], f1=scores[2], support=scores[3]))
        index += ["prf_{}_classwise".format(fold, ave)]

        for label, y_true, y_proba in zip(labels, Y_true[idx].T,
                                          Y_proba[idx].T):
            if y_true.sum():
                auc = metrics.roc_auc_score(y_true, y_proba)
                data.append(dict(label=label, fold=fold, roc_auc_score=auc))
                index += ["auc_{}_{}".format(label, fold)]

        y_true = Y_true[idx]
        y_idx = y_true.sum(axis=0) > 0
        y_true = y_true[:, y_idx]
        y_proba = Y_proba[idx][:, y_idx]
        for ave in ['micro', 'macro', 'weighted']:
            auc = metrics.roc_auc_score(y_true, y_proba, average=ave)
            data.append(dict(fold=fold, roc_auc_score=auc, average=ave))
            index += ["auc_{}_{}".format(ave, fold)]

    return pd.DataFrame.from_records(data, index=index)
