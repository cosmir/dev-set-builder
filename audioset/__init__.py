import pkg_resources

from .vggish_params import *


MODEL_PARAMS = pkg_resources.resource_filename(__name__, 'vggish_model.ckpt')
PCA_PARAMS = pkg_resources.resource_filename(__name__, 'vggish_pca_params.npz')
