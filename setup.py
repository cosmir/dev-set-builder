from setuptools import setup, find_packages

import imp

version = imp.load_source('audioset.version', 'audioset/version.py')

setup(
    name='audioset',
    version=version.version,
    description='AudioSet VGGish model',
    author='COSMIR',
    url='http://github.com/cosmir/dev-set-builder',
    download_url='http://github.com/cosmir/dev-set-builder/releases',
    packages=find_packages(),
    package_data={'': ['audioset/.model/vggish_model.cpkt',
                       'audioset/.model/vggish_pca_params.npz']},
    long_description='AudioSet VGGish model',
    classifiers=[
        "License :: OSI Approved :: ?",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='audio music vggish',
    license='?',
    install_requires=[
        'pandas',
        'numpy>=1.12',
        'scipy',
        'resampy',
        'tensorflow',
        'tqdm',
        'librosa',
        'jams>=0.3',
        'dask>=0.15.0',
        'tqdm'
    ],
    extras_require={},
    scripts=['scripts/featurefy.py']
)
