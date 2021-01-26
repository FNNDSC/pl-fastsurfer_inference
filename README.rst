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

N.B. This plug-in is a GPU efficient plug-in. It takes <1 minute to complete inference on a single brain.mgz file.
     In case a GPU is not available, a system with minimum 24GB RAM is required to run this plug-in. It takes about 90 minutes to complete inference on
     one subject on a CPU

Citations
---------

This plug-in uses the FastSurfer application built by Leonie Henschel, Sailesh Conjeti, Santiago Estrada, Kersten Diers, Bruce Fischl & Martin Reuter

The link to the publication can be found here : https://www.sciencedirect.com/science/article/pii/S1053811920304985

The source code of FastSurfer is available on Github: https://github.com/Deep-MI/FastSurfer.



Synopsis
--------

.. code::

    python fastsurfer_inference.py                                           \
        [-v <level>] [--verbosity <level>]                          \
        [--version]                                                 \
        [--man]                                                     \
        [--meta]                                                    \
        [--multi <dir containing mgz files of multiple subjects>]   \
        [--in_name <name of the i/p mgz file>]                      \
        [--out_name <name of the o/p segmented mgz file>]           \
        [--order <order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)>] \
        [--tag/-t <Search tag to process only certain subjects. If a single image should be analyzed, set the '
                           'tag with its id. Default: processes all.'>]\
        [--log <name of the log file>]                              \
        [--network_sagittal_path <path to pre-trained weights of sagittal network>] \
        [--network_coronal_path <pre-trained weights of coronal network>] \
        [--network_axial_path <pre-trained weights of axial network>] \
        [--clean]                                                    \
        [--no_cuda]                                                  \
        [--batch_size <Batch size for inference. Default: 8>]        \
        [--simple_run]                                               \
        [--run_parallel]                                             \
        [--copyInputImage]                                           \
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
    
    [--multi <dir containing mgz files of multiple subjects>]   \
    If this argument is selected then the plug-in can process multiple subjects sequentially in a single run.
    
    [--in_name <name of the i/p mgz file>]                      \
    The name of the raw .mgz file of a subject. The default value is brain.mgz
    
    [--out_name <name of the o/p segmented mgz file>]           \
    The name of the o/p or segmented mgz file. Default name is aparc.DKTatlas+aseg.deep.mgz
    If a separate subfolder is desired (e.g. FS conform, add it to the name: '
                           'mri/aparc.DKTatlas+aseg.deep.mgz)')
    
    [--order <order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)>] \
    
    [--tag/-t <Search tag to process only certain subjects. If a single image should be analyzed, set the '
                           'tag with its id. Default: processes all.'>]\
                           
    [--log <name of the log file>]                              \
    The name of the log file containing inference info. Default value is `deep-seg.log`
    
    [--network_sagittal_path <path to pre-trained weights of sagittal network>] \
    The path where a trained sagittal network resides. Default value is '../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl'
    
    [--network_coronal_path <pre-trained weights of coronal network>] \
    The path where a trained sagittal network resides. Default value is '../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl'
    
    [--network_axial_path <pre-trained weights of axial network>] \
    The path where a trained sagittal network resides. Default value is '../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl'
    
    [--clean] \
    Flag to clean up segmentation
    
    [--no_cuda] \
    The plug-in uses CPU for computation if this argument is specified. Approximate time taken is 1:30 hrs per subject
    
    [--batch_size <Batch size for inference. Default: 8>] \
    
    [--simple_run <Simplified run: only analyse one given image specified by --in_name (output: --out_name).>] \
    Need to specify absolute path to both --in_name and --out_name if this option is chosen.
    
    [--run_parallel]                \
    If specified and multiple GPUs exists, inference runs parallely on multiple GPUs. Default mode is false
    
    [--copyInputImage]
    If specified, copies input mgz file to o/p dir. Default value is false

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
            --tag . /incoming /outgoing

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-fastsurfer_inference fastsurfer_inference.py                        \
            --man                                                       \
            /incoming /outgoing

Examples
--------

This is just a quick and dirty way to get the plug-in working. Remember, the input directory should have the below structure for `--multi` feature to work

.. code:: bash

   -> inputdir
       -> Subjects
           -> Subject1
              -> brain.mgz
           -> Subject2
           -> Subject3
           .
           .
           .
           -> SubjectN
       
       
Running the plug-in on GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

(Note: the parameter ```--gpus all``` is not required. If however this plug-in fails to access the GPU, use the parameters as mentioned below)


To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

.. code:: bash

   docker run --rm --gpus all -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-fastsurfer_inference fastsurfer_inference.py     \
            --t Subject1 --in_name brain.mgz                             \
            /incoming /outgoing

The output file will be saved as /outgoing/Subject1/aparc.DKTatlas+aseg.deep.mgz



