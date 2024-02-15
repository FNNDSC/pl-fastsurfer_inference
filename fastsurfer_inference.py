#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from chris_plugin import chris_plugin, PathMapper
from    loguru              import logger
from    pftag               import pftag
from    pflog               import pflog
from    datetime            import datetime
from typing import List
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

__version__ = '1.3.7'

DISPLAY_TITLE = r"""
  __          _                   __          _        __                             
 / _|        | |                 / _|        (_)      / _|                            
| |_ __ _ ___| |_ ___ _   _ _ __| |_ ___ _ __ _ _ __ | |_ ___ _ __ ___ _ __   ___ ___ 
|  _/ _` / __| __/ __| | | | '__|  _/ _ \ '__| | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \
| || (_| \__ \ |_\__ \ |_| | |  | ||  __/ |  | | | | | ||  __/ | |  __/ | | | (_|  __/
|_| \__,_|___/\__|___/\__,_|_|  |_| \___|_|  |_|_| |_|_| \___|_|  \___|_| |_|\___\___|
                                        ______                                        
                                       |______|                                       
""" + "\t\t\t\t -- version " + __version__ + " --\n\n"


parser = ArgumentParser(description='A ChRIS plugin to run FastSurfer for creating segmentation and surfaces',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--fs_license',
                  dest='fs_license',
                  type=str,
                  help=" Path to FreeSurfer license key file.",
                  default="/fastsurfer/fs_license.txt")
parser.add_argument('--t1',
                  type=str,
                  dest='t1',
                  help='name of the input (raw) file to process (default: brain.mgz)',
                  default='brain.mgz')
parser.add_argument('--sid',
                  dest='sid',
                  type=str,
                  default="subject-000",
                  help='Subject ID for directory inside output dir to be created')
parser.add_argument('--seg',
                  dest='seg',
                  type=str,
                  default="",
                  help='Path to segmented file inside inputdir')
parser.add_argument('--seg_log',
                  dest='seg_log',
                  type=str,
                  help='name of logfile (default: deep-seg.log)',
                  default='deep-seg.log')
parser.add_argument('--clean_seg',
                  dest='clean_seg',
                  action  = 'store_true',
                  default=False,
                  help="if specified, clean up segmentation")
parser.add_argument('--no_cuda',
                  dest='no_cuda',
                  action  = 'store_true',
                  default=False,
                  help='if specified, do not use GPU')
parser.add_argument('--batch',
                  dest='batch',
                  type=str,
                  default='8',
                  help="batch size for inference (default: 8")
parser.add_argument('--simple_run',
                  dest='simple_run',
                  default=False,
                  action  = 'store_true',
                  help='simplified run: only analyze one subject')
parser.add_argument('--parallel',
                  dest='parallel',
                  action  = 'store_true',
                  default=False,
                  help='Run both hemispheres in parallel')
parser.add_argument('--seg_only',
                  dest='seg_only',
                  action  = 'store_true',
                  default=False,
                  help='Run only FastSurferCNN (generate segmentation, do not run surface pipeline)')
parser.add_argument('--surf_only',
                  dest='surf_only',
                  action  = 'store_true',
                  default=False,
                  help='Run surface pipeline only. The segmentation input has to exist already in this case.')
parser.add_argument('--seg_with_cc_only',
                  dest='seg_with_cc_only',
                  action  = 'store_true',
                  default=False,
                  help=' Run FastSurferCNN (generate segmentation) and recon_surf until corpus callosum'
                       ' (CC) is added in (no surface models will be created in this case!)')
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
    min_memory_limit='16Gi',    # supported units: Mi, Gi
    min_cpu_limit='8000m',       # millicores, e.g. "1000m" = 1 CPU core
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
    l_cli_params =[FS_SCRIPT]
    l_cli_params.extend(get_param_list(options))
    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=f"**/{options.t1}", fail_if_empty=False)
    for input_file, output_file in mapper:
        l_cli_params.extend(["--t1",f"{input_file}",
                             "--sd",f"{outputdir}"])
        LOG(f"Running {FS_SCRIPT} on input: {input_file.name}")
        try:
            LOG(l_cli_params)
            subprocess.call(l_cli_params)
        except Exception as ex:
            subprocess.call(FS_SCRIPT)

def get_param_list(options) -> List[str]:
    """
    A dirty hack to transform CLI params from this module to
    the FastSurfer shell script running inside the docker
    container.
    """
    list_param = []
    for k,v in options.__dict__.items():
        if  k not in ["t1", "inputdir", "outputdir"] and options.__dict__[k]:
            list_param.append(f"--{k}")
            if options.__dict__[k]!=True:
                list_param.append(v)

    return list_param



if __name__ == '__main__':
    main()