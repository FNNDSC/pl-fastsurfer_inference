from setuptools import setup
import re

_version_re = re.compile(r"(?<=^__version__ = (\"|'))(.+)(?=\"|')")

def get_version(rel_path: str) -> str:
    """
    Searches for the ``__version__ = `` line in a source code file.

    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    """
    with open(rel_path, 'r') as f:
        matches = map(_version_re.search, f)
        filtered = filter(lambda m: m is not None, matches)
        version = next(filtered, None)
        if version is None:
            raise RuntimeError(f'Could not find __version__ in {rel_path}')
        return version.group(0)

setup(
    name             = 'fastsurfer_inference',
    version          =  get_version('fastsurfer_inference.py'),
    description      = 'An app to efficiently perform cortical parcellation and segmentation on raw brain MRI images',
    author           = 'Martin Reuter (FastSurfer), Sandip Samal (FNNDSC) (sandip.samal@childrens.harvard.edu)',
    author_email     = 'dev@babyMRI.org',
    url              = 'http://wiki',
    py_modules       = ['fastsurfer_inference'],
    install_requires = ['chris_plugin'],
    test_suite       = 'nose.collector',
    tests_require    = ['nose'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.6',
    entry_points     = {
        'console_scripts': [
            'fastsurfer_inference = fastsurfer_inference:main'
            ]
        }
)
