from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.rst')) as f:
    readme = f.read()

setup(
    name             = 'fastsurfer_inference',
    version          = '1.3.0',
    description      = 'An app to efficiently perform cortical parcellation and segmentation on raw brain MRI images',
    long_description = readme,
    author           = 'Martin Reuter (FastSurfer), Sandip Samal (FNNDSC) (sandip.samal@childrens.harvard.edu)',
    author_email     = 'dev@babyMRI.org',
    url              = 'http://wiki',
    packages         = ['fastsurfer_inference','data_loader','models'],
    install_requires = ['chrisapp'],
    test_suite       = 'nose.collector',
    tests_require    = ['nose'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.6',
    entry_points     = {
        'console_scripts': [
            'fastsurfer_inference = fastsurfer_inference.__main__:main'
            ]
        }
)
