#!/usr/bin/env python
#
# fastsurfer_inference ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# IMPORTS
import optparse
import nibabel as nib
import numpy as np
import time
import sys
import glob
import os
import os.path as op
import logging
import torch
import torch.nn as nn
import shutil

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, utils

from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.measure import label, regionprops
from skimage.measure import label

from collections import OrderedDict
from os import makedirs

from data_loader.load_neuroimaging_data import OrigDataThickSlices
from data_loader.load_neuroimaging_data import map_label2aparc_aseg
from data_loader.load_neuroimaging_data import map_prediction_sagittal2full
from data_loader.load_neuroimaging_data import get_largest_cc
from data_loader.load_neuroimaging_data import load_and_conform_image

from data_loader.augmentation import ToTensorTest

from models.networks import FastSurferCNN

sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp


Gstr_title = """

  __          _                   __          _        __                             
 / _|        | |                 / _|        (_)      / _|                            
| |_ __ _ ___| |_ ___ _   _ _ __| |_ ___ _ __ _ _ __ | |_ ___ _ __ ___ _ __   ___ ___ 
|  _/ _` / __| __/ __| | | | '__|  _/ _ \ '__| | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \ 
| || (_| \__ \ |_\__ \ |_| | |  | ||  __/ |  | | | | | ||  __/ | |  __/ | | | (_|  __/
|_| \__,_|___/\__|___/\__,_|_|  |_| \___|_|  |_|_| |_|_| \___|_|  \___|_| |_|\___\___|
                                        ______
                                       |______|

"""

Gstr_synopsis = """

    NAME

       fastsurfer_inference.py

    SYNOPSIS

        python fastsurfer_inference.py                                      \\
                                    [--subjectDir <subjectDir>]             \\
                                    [--subject <subjectToProcess>]          \\
                                    [--in_name <inputFileToProcess>]        \\
                                    [--out_name <segmentedFile]             \\
                                    [--order <interpolation>]               \\
                                    [--log <logFile>]                       \\
                                    [--clean]                               \\
                                    [--no_cuda]                             \\
                                    [--batch_size <batchSizePerInference]   \\
                                    [--simple_run]                          \\
                                    [--run_parallel]                        \\
                                    [--copyInputFiles <seachKeyword>]       \\
                                    [-v <level>] [--verbosity <level>]      \\
                                    [--version]                             \\
                                    [--man]                                 \\
                                    [--meta]                                \\
                                    <inputDir>
                                    <outputDir>

    BRIEF EXAMPLE

        * Bare bones execution:

            Assume that the ``in`` directory contains a single subject
            ``subdir``, which in turn contains single file, called
            ``brain.mgz``:

            mkdir out && chmod 777 out
            docker run --rm                                                 \\
                        -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \\
                        fnndsc/pl-fastsurfer_inference                      \\
                        fastsurfer_inference.py                             \\
                        /incoming /outgoing

    DESCRIPTION

        ``fastsurfer_inference.py`` is a ChRIS plugin constructed around the
        FastSurfer application developed by the "Deep Medical Imaging Lab"
        (https://deep-mi.org). For the publication describing this method, see

            Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M.
            "FastSurfer - A fast and accurate deep learning based neuroimaging
            pipeline." NeuroImage. 2020.

            https://arxiv.org/pdf/2009.04392.pdf
            https://deep-mi.org/static/pub/ewert_2020.bib

        This specific plugin wraps around many of the FastSurfer CLI in
        addition to providing ChRIS plugin-specific CLI. It's primary purpose
        is to segment and input brain image file (in FreeSurfer ``mgz`` format),
        creating in turn as output a labeled ``mgz`` format file.

    ARGS

        [--subjectDir <subjectDir>]
        By default, the <subjectDir> is assumed to be the <inputDir>. However,
        the <subjectDir> can be nested relative to the <inputDir>, and can thus
        be specified with this flag.

        The <subjectDir> is assumed by default to contain one level of sub
        directory, and these sub dirs, considered the ``subjects``, each contain
        a single ``mgz`` to process.

        [--subject <subjectToProcess>]
        This can denote a sub-set of subject(s) (i.e. sub directory within the
        <subjectDir>). The <subjectToProcess> is "globbed", so an expression
        like ``--subject 10*`` would process all ``subjects`` starting with the
        text string ``10``. Note to protect from shell expansion of wildcard
        characters, the argument should be protected in single quotes.

        [--in_name <inputFileToProcess>]
        The name of the raw ``.mgz`` file of a subject. The default value is
        ``brain.mgz``. The full path to the <inputFileToProcess> is constructed
        by concatenating

                ``<inputDir>/<subjectDir>/<subject>/<inputFileToProcess>``

        [--out_name <segmentedFile]
        The name of the output or segmented ``mgz`` file. Default name is

                            ``aparc.DKTatlas+aseg.deep.mgz``

        [--order <interpolation>]
        The order of interpolation:

                            0 = nearest
                            1 = linear (default)
                            2 = quadratic
                            3 = cubic

        [--log <logFile>]
        The name of the log file containing inference info. Default value is

                            ``deep-seg.log``

        [--clean]
        If specified, clean the segmentation.

        [--no_cuda]
        If specified, run on CPU, not GPU. Depending on CPU/GPU, your apparent
        mileage will vary, but expect orders longer time than compared to a
        GPU.

        For example, in informal testing, GPU takes about a minute per
        subject, while CPU approximately 1.5 hours per subject!

        [--batch_size <batchSizePerInference]
        Batch size per inference. Default is 8.

        [--simple_run]
        Simplified run: only analyse one given image specified by ``--in_name``
        (output: ``--out_name``). Note that you need to specify absolute path
        to both ``--in_name`` and ``--out_name`` if this option is chosen.

        [--run_parallel]
        If multiple GPUs are present to the docker container, enable parallel
        computation on multiple GPUs with an inference run.

        [--copyInputFiles <searchKeyword>]
        If specified, copies the input files matching the keywords to output dir. This can be
        useful to create an easy association between a given input files and the
        segmented output.

        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.

        [--version]
        If specified, print version number.

        [--man]
        If specified, print (this) man page.

        [--meta]
        If specified, print plugin meta data.

"""


class Fastsurfer_inference(ChrisApp):
    """
    An app to efficiently perform cortical parcellation and segmentation
    on raw brain MRI images, using FastSurfer developed by the
    Deep Medical Imaging Lab (https://deep-mi.org).
    """
    AUTHORS                 = 'Martin Reuter (FastSurfer), Sandip Samal (FNNDSC) (sandip.samal@childrens.harvard.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'An inference app of FastSurfer'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app to efficiently perform cortical parcellation and segmentation on raw brain MRI images'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '1.0.15'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '30000m' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '30000Mi' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        # Requiered options
        # 1. Directory information (where to read from, where to write to)
        self.add_argument(  '--subjectDir',
                            dest        = 'subjectDir',
                            type        = str,
                            optional    = True,
                            help        = "directory (relative to <inputDir>) of subjects to process",
                            default     = "")

        # 2. Options for the MRI volumes
        # (name of in and output, order of interpolation if not conformed)
        self.add_argument(  '--in_name', '-i',
                            type        = str,
                            dest        = 'iname',
                            help        = 'name of the input (raw) file to process (default: brain.mgz)',
                            optional    = True,
                            default     = 'brain.mgz')
        self.add_argument(  '--out_name', '-o',
                            dest        = 'oname',
                            type        = str,
                            optional    = True,
                            default     = 'aparc.DKTatlas+aseg.deep.mgz',
                            help        = 'name of the output segmented file')
        self.add_argument(  '--order',
                            dest        = 'order',
                            type        = int,
                            default     = 1,
                            optional    = True,
                            help        = "interpolation order")

        # 3. Options for log-file and search-tag
        self.add_argument(  '--subject',
                            dest        = 'subject',
                            type        = str,
                            default     = "*",
                            optional    = True,
                            help        = 'subject(s) to process. This expression is globbed.')
        self.add_argument(  '--log',
                            dest        = 'logfile',
                            type        = str,
                            optional    = True,
                            help        = 'name of logfile (default: deep-seg.log)',
                            default     = 'deep-seg.log')

        # 4. Pre-trained weights -- NB NB NB -- currently (Jan 2021) these CANNOT
        # be set by an enduser. These weight files are RELATIVE/INTERNAL to the
        # container
        self.add_argument(  '--network_sagittal_path',
                            dest        = 'network_sagittal_path',
                            type        = str,
                            optional    = True,
                            help        = "path to pre-trained sagittal network weights",
                            default     = '../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
        self.add_argument(  '--network_coronal_path',
                            dest        = 'network_coronal_path',
                            type        = str,
                            optional    = True,
                            help        = "path to pre-trained coronal network weights",
                            default     = '../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
        self.add_argument(  '--network_axial_path',
                            dest        = 'network_axial_path',
                            type        = str,
                            optional    = True,
                            help        = "path to pre-trained axial network weights",
                            default     = '../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')

        # 5. Clean up and GPU/CPU options (disable cuda, change batchsize)
        self.add_argument(  '--clean',
                            dest        = 'cleanup',
                            type        = bool,
                            optional    = True,
                            default     = True,
                            help        = "if specified, clean up segmentation",
                            action      = 'store_true')
        self.add_argument(  '--no_cuda',
                            dest        = 'no_cuda',
                            action      = 'store_true',
                            type        = bool,
                            optional    = True,
                            default     = False,
                            help        = 'if specified, do not use GPU')
        self.add_argument(  '--batch_size',
                            dest        = 'batch_size',
                            type        = int,
                            default     = 8,
                            optional    = True,
                            help        = "batch size for inference (default: 8")
        self.add_argument(  '--simple_run',
                            dest        = 'simple_run',
                            action      = 'store_true',
                            optional    = True,
                            default     = False,
                            type        = bool,
                            help        = 'simplified run: only analyze one subject')

        # Adding check to parallel processing, default = false
        self.add_argument(  '--run_parallel',
                            dest        = 'run_parallel',
                            type        = bool,
                            action      = 'store_true',
                            optional    = True ,
                            default     = False,
                            help        = 'if specified, allows for execute on multiple GPUs')

        self.add_argument(  '--copyInputFiles',
                            dest        = 'copyInputFiles',
                            type        = str,
                            default     = "",
                            optional    = True,
                            help        = "if specified, copy i/p files matching the input regex to o/p dir")

    def fast_surfer_cnn(self,img_filename, save_as, logger, args):
        """
        Cortical parcellation of single image
        :param str img_filename: name of image file
        :param parser.Options args: Arguments (passed via command line) to set up networks
            * args.network_sagittal_path: path to sagittal checkpoint (stored pretrained network)
            * args.network_coronal_path: path to coronal checkpoint (stored pretrained network)
            * args.network_axial_path: path to axial checkpoint (stored pretrained network)
            * args.cleanup: Whether to clean up the segmentation (medial filter on certain labels)
            * args.no_cuda: Whether to use CUDA (GPU) or not (CPU)
        :param logging.logger logger: Logging instance info messages will be written to
        :param str save_as: name under which to save prediction.
        :return None: saves prediction to save_as
        """
        start_total = time.time()
        logger.info("Reading volume {}".format(img_filename))

        header_info, affine_info, orig_data = load_and_conform_image(img_filename, interpol=args.order)
        
        # Copy file(s) from input dir to output dir subject wise
        if args.copyInputFiles !="":
        
            # define source and destination dir
            source_dir_path = img_filename.replace(args.iname,'')
            dest_dir_path = save_as.replace(args.oname,'')
            
            # search for files using the given keyword
            found_files = glob.glob(source_dir_path+"/*%s*" %args.copyInputFiles, recursive = True)
            
            # now iteratively copy the files
            for file_path in found_files:
                logger.info("Copying files from %s" %file_path)
                shutil.copy(file_path, dest_dir_path)
            
            
        transform_test = transforms.Compose([ToTensorTest()])

        test_dataset_axial = OrigDataThickSlices(img_filename, orig_data, transforms=transform_test, plane='Axial')
        test_dataset_sagittal = OrigDataThickSlices(img_filename, orig_data, transforms=transform_test, plane='Sagittal')
        test_dataset_coronal = OrigDataThickSlices(img_filename, orig_data, transforms=transform_test, plane='Coronal')

        start = time.time()

        test_data_loader = DataLoader(dataset=test_dataset_axial, batch_size=args.batch_size, shuffle=False)

        # Axial View Testing
        params_network = {'num_channels': 7, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1, 'pool': 2,
                      'stride_pool': 2, 'num_classes': 79, 'kernel_c': 1, 'kernel_d': 1}

        # Select the model
        model = FastSurferCNN(params_network)

        # Put it onto the GPU or CPU
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info("Cuda available: {}, # Available GPUS: {}, "
                    "Cuda user disabled (--no_cuda flag): {}, --> Using device: {}".format(torch.cuda.is_available(),
                                                                                       torch.cuda.device_count(),
                                                                                       args.no_cuda, device))

        if use_cuda and torch.cuda.device_count() > 1 and args.run_parallel:
            model = nn.DataParallel(model)
            model_parallel = True
        else:
            model_parallel = False

        model.to(device)

        # Set up state dict (remapping of names, if not multiple GPUs/CPUs)
        logger.info("Loading Axial Net from {}".format(args.network_axial_path))

        model_state = torch.load(args.network_axial_path, map_location=device)
        new_state_dict = OrderedDict()

        for k, v in model_state["model_state_dict"].items():

            if k[:7] == "module." and not model_parallel:
                new_state_dict[k[7:]] = v

            elif k[:7] != "module." and model_parallel:
                new_state_dict["module." + k] = v

            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

        model.eval()
        prediction_probability_axial = torch.zeros((256, params_network["num_classes"], 256, 256), dtype=torch.float)

        logger.info("Axial model loaded.")
        with torch.no_grad():

            start_index = 0
            for batch_idx, sample_batch in enumerate(test_data_loader):
                images_batch = Variable(sample_batch["image"])

                if use_cuda:
                    images_batch = images_batch.cuda()

                temp = model(images_batch)

                prediction_probability_axial[start_index:start_index + temp.shape[0]] = temp.cpu()
                start_index += temp.shape[0]
                logger.info("--->Batch {} Axial Testing Done.".format(batch_idx))

        logger.info("Axial View Tested in {:0.4f} seconds".format(time.time() - start))

        # Coronal View Testing
        start = time.time()

        test_data_loader = DataLoader(dataset=test_dataset_coronal, batch_size=args.batch_size, shuffle=False)

        params_network = {'num_channels': 7, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1, 'pool': 2,
                      'stride_pool': 2, 'num_classes': 79, 'kernel_c': 1, 'kernel_d': 1}

        # Select the model

        model = FastSurferCNN(params_network)

        # Put it onto the GPU or CPU
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda and torch.cuda.device_count() > 1 and args.run_parallel:
            model = nn.DataParallel(model)
            model_parallel = True
        else:
            model_parallel = False

        model.to(device)

        # Set up new state dict (remapping of names, if not multiple GPUs/CPUs)
        logger.info("Loading Coronal Net from {}".format(args.network_coronal_path))

        model_state = torch.load(args.network_coronal_path, map_location=device)
        new_state_dict = OrderedDict()

        for k, v in model_state["model_state_dict"].items():

            if k[:7] == "module." and not model_parallel:
                new_state_dict[k[7:]] = v

            elif k[:7] != "module." and model_parallel:
                new_state_dict["module." + k] = v

            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

        model.eval()
        prediction_probability_coronal = torch.zeros((256, params_network["num_classes"], 256, 256), dtype=torch.float)

        logger.info("Coronal model loaded.")
        start_index = 0
        with torch.no_grad():

            for batch_idx, sample_batch in enumerate(test_data_loader):

                images_batch = Variable(sample_batch["image"])

                if use_cuda:
                    images_batch = images_batch.cuda()

                temp = model(images_batch)

                prediction_probability_coronal[start_index:start_index + temp.shape[0]] = temp.cpu()
                start_index += temp.shape[0]
                logger.info("--->Batch {} Coronal Testing Done.".format(batch_idx))

        logger.info("Coronal View Tested in {:0.4f} seconds".format(time.time() - start))

        start = time.time()

        test_data_loader = DataLoader(dataset=test_dataset_sagittal, batch_size=args.batch_size, shuffle=False)

        params_network = {'num_channels': 7, 'num_filters': 64, 'kernel_h': 5, 'kernel_w': 5, 'stride_conv': 1, 'pool': 2,
                      'stride_pool': 2, 'num_classes': 51, 'kernel_c': 1, 'kernel_d': 1}

        # Select the model
        model = FastSurferCNN(params_network)

        # Put it onto the GPU or CPU
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda and torch.cuda.device_count() > 1 and args.run_parallel:
            model = nn.DataParallel(model)
            model_parallel = True
        else:
            model_parallel = False

        model.to(device)

        # Set up new state dict (remapping of names, if not multiple GPUs/CPUs)
        logger.info("Loading Sagittal Net from {}".format(args.network_sagittal_path))

        model_state = torch.load(args.network_sagittal_path, map_location=device)
        new_state_dict = OrderedDict()

        for k, v in model_state["model_state_dict"].items():

            if k[:7] == "module." and not model_parallel:
                new_state_dict[k[7:]] = v

            elif k[:7] != "module." and model_parallel:
                new_state_dict["module." + k] = v

            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

        model.eval()
        prediction_probability_sagittal = torch.zeros((256, params_network["num_classes"], 256, 256), dtype=torch.float)

        start_index = 0
        with torch.no_grad():

            for batch_idx, sample_batch in enumerate(test_data_loader):

                images_batch = Variable(sample_batch["image"])

                if use_cuda:
                    images_batch = images_batch.cuda()

                temp = model(images_batch)

                prediction_probability_sagittal[start_index:start_index + temp.shape[0]] = temp.cpu()
                start_index += temp.shape[0]
                logger.info("--->Batch {} Sagittal Testing Done.".format(batch_idx))

        prediction_probability_sagittal = map_prediction_sagittal2full(prediction_probability_sagittal)

        logger.info("Sagittal View Tested in {:0.4f} seconds".format(time.time() - start))

        del model, test_dataset_axial, test_dataset_coronal, test_dataset_sagittal, test_data_loader

        start = time.time()

        # Start View Aggregation: change from N,C,X,Y to coronal view with C in last dimension = H,W,D,C
        prediction_probability_axial = prediction_probability_axial.permute(3, 0, 2, 1)
        prediction_probability_coronal = prediction_probability_coronal.permute(2, 3, 0, 1)
        prediction_probability_sagittal = prediction_probability_sagittal.permute(0, 3, 2, 1)

        intermediate_img = torch.add(prediction_probability_axial, prediction_probability_coronal)
        del prediction_probability_axial, prediction_probability_coronal

        _, prediction_image = torch.max(torch.add(torch.mul(intermediate_img, 0.4),
                                              torch.mul(prediction_probability_sagittal, 0.2)), 3)

        del prediction_probability_sagittal, intermediate_img

        prediction_image = prediction_image.numpy()

        end = time.time() - start
        logger.info("View Aggregation finished in {:0.4f} seconds.".format(end))

        prediction_image = map_label2aparc_aseg(prediction_image)

        # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
        rh_wm = get_largest_cc(prediction_image == 41)
        lh_wm = get_largest_cc(prediction_image == 2)
        rh_wm = regionprops(label(rh_wm, background=0))
        lh_wm = regionprops(label(lh_wm, background=0))
        centroid_rh = np.asarray(rh_wm[0].centroid)
        centroid_lh = np.asarray(lh_wm[0].centroid)

        labels_list = np.array([1003, 1006, 1007, 1008, 1009, 1011,
                            1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

        for label_current in labels_list:

            label_img = label(prediction_image == label_current, connectivity=3, background=0)

            for region in regionprops(label_img):

                if region.label != 0:  # To avoid background

                    if np.linalg.norm(np.asarray(region.centroid) - centroid_rh) < np.linalg.norm(
                            np.asarray(region.centroid) - centroid_lh):
                        mask = label_img == region.label
                        prediction_image[mask] = label_current + 1000

        # Quick Fixes for overlapping classes
        aseg_lh = gaussian_filter(1000 * np.asarray(prediction_image == 2, dtype=np.float), sigma=3)
        aseg_rh = gaussian_filter(1000 * np.asarray(prediction_image == 41, dtype=np.float), sigma=3)

        lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3),
                            axis=3)

        # Problematic classes: 1026, 1011, 1029, 1019
        for prob_class_lh in [1011, 1019, 1026, 1029]:
            prob_class_rh = prob_class_lh + 1000
            mask_lh = ((prediction_image == prob_class_lh) | (prediction_image == prob_class_rh)) & (lh_rh_split == 0)
            mask_rh = ((prediction_image == prob_class_lh) | (prediction_image == prob_class_rh)) & (lh_rh_split == 1)

            prediction_image[mask_lh] = prob_class_lh
            prediction_image[mask_rh] = prob_class_rh

        # Clean-Up
        if args.cleanup is True:

            labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                      15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                      46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                      77, 1026, 2026]

            start = time.time()
            prediction_image_medfilt = median_filter(prediction_image, size=(3, 3, 3))
            mask = np.zeros_like(prediction_image)
            tolerance = 25

            for current_label in labels:
                current_class = (prediction_image == current_label)
                label_image = label(current_class, connectivity=3)

                for region in regionprops(label_image):

                    if region.area <= tolerance:
                        mask_label = (label_image == region.label)
                        mask[mask_label] = 1

            prediction_image[mask == 1] = prediction_image_medfilt[mask == 1]
            logger.info("Segmentation Cleaned up in {:0.4f} seconds.".format(time.time() - start))

        # Saving image
        header_info.set_data_dtype(np.int16)
        mapped_aseg_img = nib.MGHImage(prediction_image, affine_info, header_info)
        mapped_aseg_img.to_filename(save_as)

        logger.info("Saving Segmentation to {}".format(save_as))
        logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))



    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())

        # Output the space of CLI
        d_options = vars(options)
        for k,v in d_options.items():
            print("%20s: %-40s" % (k, v))
        print("")

        # Command Line options and error checking done here
        # options = options_parse()

        # Set up the logger
        logger = logging.getLogger("eval")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(stream=sys.stdout))

        if options.simple_run:

            # Check if output subject directory exists and create it otherwise
            sub_dir, out = op.split(options.oname)

            if not op.exists(sub_dir):
                makedirs(sub_dir)

            fast_surfer_cnn(options.iname, options.oname, logger, options)

        else:

            # Prepare subject list to be processed
            if options.subjectDir != "":
                search_path = op.join(  options.inputdir,
                                        options.subjectDir,
                                        options.subject)
                subject_directories = glob.glob(search_path)

            else:
                search_path = op.join(options.inputdir, options.subject)
                subject_directories = glob.glob(search_path)

            # Report number of subjects to be processed and loop over them
            data_set_size = len(subject_directories)
            logger.info("Total Dataset Size is {}".format(data_set_size))

            for current_subject in subject_directories:

                subject = current_subject.split("/")[-1]

                # Define volume to process, log-file and name under which final prediction will be saved


                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.outputdir, subject, options.logfile)
                save_file_name = op.join(options.outputdir, subject, options.oname)


                logger.info("Running Fast Surfer on {}".format(subject))

                # Check if output subject directory exists and create it otherwise
                sub_dir, out = op.split(save_file_name)

                if not op.exists(sub_dir):
                    makedirs(sub_dir)

                # Prepare the log-file (logging to File in subject directory)
                fh = logging.FileHandler(logfile, mode='w')
                logger.addHandler(fh)

                # Run network
                self.fast_surfer_cnn(invol, save_file_name, logger, options)

                logger.removeHandler(fh)
                fh.close()

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Fastsurfer_inference()
    chris_app.launch()
