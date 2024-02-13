#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin, PathMapper
from    loguru              import logger
from    pftag               import pftag
from    pflog               import pflog
from    datetime            import datetime
import sys
import os
import subprocess
LOG             = logger.debug

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> │ "
    "<level>{level: <5}</level> │ "
    "<yellow>{name: >28}</yellow>::"
    "<cyan>{function: <30}</cyan> @"
    "<cyan>{line: <4}</cyan> ║ "
    "<level>{message}</level>"
)
logger.remove()
logger.opt(colors = True)
logger.add(sys.stderr, format=logger_format)

__version__ = '1.3.2'

DISPLAY_TITLE = r"""
  __          _                   __          _        __                             
 / _|        | |                 / _|        (_)      / _|                            
| |_ __ _ ___| |_ ___ _   _ _ __| |_ ___ _ __ _ _ __ | |_ ___ _ __ ___ _ __   ___ ___ 
|  _/ _` / __| __/ __| | | | '__|  _/ _ \ '__| | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \
| || (_| \__ \ |_\__ \ |_| | |  | ||  __/ |  | | | | | ||  __/ | |  __/ | | | (_|  __/
|_| \__,_|___/\__|___/\__,_|_|  |_| \___|_|  |_|_| |_|_| \___|_|  \___|_| |_|\___\___|
                                        ______                                        
                                       |______|                                       
""" + "\t\t -- version " + __version__ + " --\n\n"


parser = ArgumentParser(description='A ChRIS plugin to send DICOMs to a remote PACS store',
                        formatter_class=ArgumentDefaultsHelpFormatter)
# Required options
# 1. Directory information (where to read from, where to write to)
parser.add_argument('--subjectDir',"--sd",
                  dest='subjectDir',
                  type=str,
                  help="directory (relative to <inputDir>) of subjects to process",
                  default="")

# 2. Options for the MRI volumes
# (name of in and output, order of interpolation if not conformed)
parser.add_argument('--iname', '--t1',
                  type=str,
                  dest='iname',
                  help='name of the input (raw) file to process (default: brain.mgz)',
                  default='brain.mgz')
parser.add_argument('--out_name', '--seg',
                  dest='oname',
                  type=str,
                  default='aparc.DKTatlas+aseg.deep.mgz',
                  help='name of the output segmented file')
parser.add_argument('--order',
                  dest='order',
                  type=int,
                  default=1,
                  help="interpolation order")

# 3. Options for log-file and search-tag
parser.add_argument('--subject',
                  dest='subject',
                  type=str,
                  default="*",
                  help='subject(s) to process. This expression is globbed.')
parser.add_argument('--log',
                  dest='logfile',
                  type=str,
                  help='name of logfile (default: deep-seg.log)',
                  default='deep-seg.log')

# 4. Pre-trained weights -- NB NB NB -- currently (Jan 2021) these CANNOT
# be set by an enduser. These weight files are RELATIVE/INTERNAL to the
# container
parser.add_argument('--network_sagittal_path',
                  dest='network_sagittal_path',
                  type=str,
                  help="path to pre-trained sagittal network weights",
                  default='./checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
parser.add_argument('--network_coronal_path',
                  dest='network_coronal_path',
                  type=str,
                  help="path to pre-trained coronal network weights",
                  default='./checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
parser.add_argument('--network_axial_path',
                  dest='network_axial_path',
                  type=str,
                  help="path to pre-trained axial network weights",
                  default='./checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')

# 5. Clean up and GPU/CPU options (disable cuda, change batchsize)
parser.add_argument('--clean',
                  dest='cleanup',
                  type=bool,
                  default=True,
                  help="if specified, clean up segmentation")
parser.add_argument('--no_cuda',
                  dest='no_cuda',
                  type=bool,
                  default=False,
                  help='if specified, do not use GPU')
parser.add_argument('--batch_size',
                  dest='batch_size',
                  type=int,
                  default=8,
                  help="batch size for inference (default: 8")
parser.add_argument('--simple_run',
                  dest='simple_run',
                  default=False,
                  type=bool,
                  help='simplified run: only analyze one subject')

# Adding check to parallel processing, default = false
parser.add_argument('--run_parallel',
                  dest='run_parallel',
                  type=bool,
                  default=False,
                  help='if specified, allows for execute on multiple GPUs')

parser.add_argument('--copyInputFiles',
                  dest='copyInputFiles',
                  type=str,
                  default="",
                  help="if specified, copy i/p files matching the input regex to o/p dir")
parser.add_argument('-f', '--fileFilter', default='dcm', type=str,
                    help='input file filter glob')
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')
parser.add_argument(  '--pftelDB',
                    dest        = 'pftelDB',
                    default     = '',
                    type        = str,
                    help        = 'optional pftel server DB path')


def preamble_show(options) -> None:
    """
    Just show some preamble "noise" in the output terminal
    """

    LOG(DISPLAY_TITLE)

    LOG("plugin arguments...")
    for k, v in options.__dict__.items():
        LOG("%25s:  [%s]" % (k, v))
    LOG("")

    LOG("base environment...")
    for k, v in os.environ.items():
        LOG("%25s:  [%s]" % (k, v))
    LOG("")


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='An app to efficiently perform cortical parcellation and segmentation on raw brain MRI images',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='2Gi',    # supported units: Mi, Gi
    min_cpu_limit='4000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
@pflog.tel_logTime(
            event       = 'fastsurfer_inference',
            log         = 'Create segmentation using FastSurferCNN'
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """
    FS_SCRIPT = "/fastsurfer/run_fastsurfer.sh"
    preamble_show(options)
    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/{options.iname}", fail_if_empty=False)
    for input_file, output_file in mapper:
        l_cli_params = [FS_SCRIPT]
        l_cli_params.extend(["--sd", outputdir,
                             "--t1", f"{input_file}",
                             "--sid", f"{options.subject}",
                             "--seg_only",
                             "--parallel"])
        LOG(f"Running {FS_SCRIPT} on input: {input_file.name}")
        try:
            subprocess.call(l_cli_params)
        except Exception as ex:
            subprocess.call(FS_SCRIPT)


if __name__ == '__main__':
    main()