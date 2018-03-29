import pkg_resources

from .vggish_params import *


MODEL_PARAMS = pkg_resources.resource_filename(
    __name__, '.model/vggish_model.cpkt')
PCA_PARAMS = pkg_resources.resource_filename(
    __name__, '.model/vggish_pca_params.npz')

VGG_ESTIMATOR_FILE = pkg_resources.resource_filename(
    __name__, '.vggish_estimator/openmic23-a226be/predictor.json')
VGG_ESTIMATOR_WEIGHTS = pkg_resources.resource_filename(
    __name__, '.vggish_estimator/openmic23-a226be/weights-0008.h5')
