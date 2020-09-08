pl-fastsurfer_inference
================================

.. image:: https://badge.fury.io/py/fastsurfer_inference.svg
    :target: https://badge.fury.io/py/fastsurfer_inference

.. image:: https://travis-ci.org/FNNDSC/fastsurfer_inference.svg?branch=master
    :target: https://travis-ci.org/FNNDSC/fastsurfer_inference

.. image:: https://img.shields.io/badge/python-3.5%2B-blue.svg
    :target: https://badge.fury.io/py/pl-fastsurfer_inference

.. contents:: Table of Contents


Abstract
--------

An app to efficiently perform cortical parcellation and anatomical segmentation mimicking FreeSurfer, on raw brain MRI images


Synopsis
--------

.. code::

    python fastsurfer_inference.py                                           \
        [-v <level>] [--verbosity <level>]                          \
        [--version]                                                 \
        [--man]                                                     \
        [--meta]                                                    \
        <inputDir>
        <outputDir> 

Description
-----------

``fastsurfer_inference.py`` is a ChRIS-based application that is capable of whole brain segmentation into 95 classes

TLDR
------
Just pull the docker image

.. code::

    docker pull fnndsc/pl-fastsurfer_inference

Go straight to the examples section

Arguments
---------

.. code::

    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.

    [--version]
    If specified, print version number. 
    
    [--man]
    If specified, print (this) man page.

    [--meta]
    If specified, print plugin meta data.


Run
----

This ``plugin`` can be run in two modes: natively as a python package or as a containerized docker image.

Using PyPI
~~~~~~~~~~

To run from PyPI, simply do a 

.. code:: bash

    pip install fastsurfer_inference

and run with

.. code:: bash

    fastsurfer_inference.py --man /tmp /tmp

to get inline help. The app should also understand being called with only two positional arguments

.. code:: bash

    fastsurfer_inference.py /some/input/directory /destination/directory


Using ``docker run``
~~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Now, prefix all calls with 

.. code:: bash

    docker run --rm -v $(pwd)/out:/outgoing                             \
            fnndsc/pl-fastsurfer_inference fastsurfer_inference.py                        \

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-fastsurfer_inference fastsurfer_inference.py                        \
            --man                                                       \
            /incoming /outgoing

Examples
--------

This is just a quick and dirty way to get the plug-in working. Remember, the input directory should have the below structure 

.. code:: bash

   -> inputdir
       -> Subject1
          -> brain.mgz
       -> Subject2
       -> Subject3
       .
       .
       .
       -> SubjectN
       
       
Running the plug-in on GPU (Note: the parameter ```--gpus all``` is not required. If however this plug-in fails to access the GPU, use the parameters)

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

.. code:: bash

   docker run --rm --gpus all -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-fastsurfer_inference fastsurfer_inference.py     \
            --t Subject1 --in_name brain.mgz                             \
            /incoming /outgoing

The output file will be saved as /outgoing/Subject1/aparc.DKTatlas+aseg.deep.mgz



