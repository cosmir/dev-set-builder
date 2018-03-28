from __future__ import print_function
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

instrument_all = ['bass', 'cello', 'cymbals', 'drums', 'flute', 'guitar', 'mallet_percussion', 'organ',
                  'piano', 'saxophone', 'synthesizer', 'trumpet', 'violin', 'voice']
path_project = '/your/path/to/openmic/'
path_mean_feat = path_project + 'mean_feat_vggish_subset.pickle'
path_std_feat = path_project + 'std_feat_vggish_subset.pickle'

pickle_in = open(path_mean_feat, 'rb')
mean_feat = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(path_std_feat, 'rb')
std_feat = pickle.load(pickle_in)
pickle_in.close()

for instrument in instrument_all:
    pickle_in = open(path_project + instrument + '_y.pickle', 'rb')
    y_dict = pickle.load(pickle_in)
    pickle_in.close()
    score = np.zeros(10)
    for repeat in range(10):
        X = np.zeros([1500, 256])
        y = np.zeros(1500)
        count = 0
        for indiv_key in y_dict.keys():
            X[count, :] = np.concatenate((mean_feat[indiv_key], std_feat[indiv_key]))
            y[count] = y_dict[indiv_key]
            count += 1

        rd = np.random.permutation(1500)
        X_train = X[rd[:1000], :]
        y_train = y[rd[:1000]]
        X_test = X[rd[1000:], :]
        y_test = y[rd[1000:]]

        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        clf = RandomForestClassifier(max_depth=8, random_state=0)
        clf.fit(X_train, y_train)
        score[repeat] = np.mean(clf.predict(X_test) == y_test)
    print(instrument, 'accuracy:', np.mean(score))
