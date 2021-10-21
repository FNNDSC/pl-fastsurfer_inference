from os import path
from setuptools import setup


with open(path.join(path.dirname(path.abspath(__file__)), 'README.rst')) as f:
    readme = f.read()


setup(
      name             =   'fastsurfer_inference',
      version          =   '1.2.0',
      description      =   'An app to efficiently perform cortical parcellation and segmentation on raw brain MRI images using the "fastsurfer" engine of Martin Reuter',
      long_description =   readme,
      author           =   'Sandip Samal (www.fnndsc.org)',
      author_email     =   'sandip.samal@childrens.harvard.edu',
      url              =   'https://github.com/FNNDSC/pl-fastsurfer_inference',
      packages         =   ['fastsurfer_inference'],
      install_requires =   ['chrisapp', 'pudb'],
      test_suite       =   'nose.collector',
      tests_require    =   ['nose'],
      scripts          =   ['fastsurfer_inference/fastsurfer_inference.py'],
      license          =   'MIT',
      zip_safe         =   False,
      python_requires  =   '>=3.5'
     )
