'''The main file for training a model

Usage:

'''
import os
import sys
import torch
import logging
from config.defaults import get_cfg_defaults

import argparse
import importlib
from utils.utils import mkdirs
import pdb
import random
import numpy as np
import subprocess
torch.backends.cudnn.benchmark = True


def default_argument_parser():
    """
        Create arg parser
    """
    parser = argparse.ArgumentParser("Args parser for training")
    parser.add_argument("--config", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--trainer", default="", help="The trainer")
    #

    # parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def init_seed(config):

    if config.SEED is None:

        seed = random.randrange(0, 10000)
        logging.info("No random seed is provided. Generate a seed randomly: {}".format(seed))
        config.SEED = seed
    else:
        logging.info("Using random seed from the config: {}".format(config.SEED))
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return config

def setup(args):
    """
        Perform some basic common setups at the beginning of a job, including:
    """
    cfg = get_cfg_defaults()
    trainer_lib = importlib.import_module('models.' + args.trainer)
    if 'custom_cfg' in trainer_lib.__all__:
        cfg.merge_from_other_cfg(trainer_lib.custom_cfg)

    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.DEBUG = args.debug

    cfg.OUTPUT_DIR = mkdirs(cfg.OUTPUT_DIR)



    output_dir = cfg.OUTPUT_DIR
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(filename)s-%(funcName)s-%(lineno)d:%(message)s",
                        datefmt='%a-%d %b %Y %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log'), 'a', 'utf-8'),
                                  logging.StreamHandler()]
            )

    logging.info("Config:\n" + str(cfg))
    logging.info("Command line: python " + ' '.join(sys.argv))

    cfg = init_seed(cfg)

    subprocess.check_output(['git', 'fetch'])
    git_lof_info = subprocess.check_output(['git', 'log', 'origin/master',])
    latest_master_branch_commit_id = str(git_lof_info).split('\\n')[0]
    logging.info("Git master branch head at {}".format(latest_master_branch_commit_id))

    config_saved_path = os.path.join(output_dir, "train_config.yaml")
    with open(config_saved_path, "w") as f:
        f.write(cfg.dump())
        logging.info("Full config saved to {}".format(config_saved_path))

    cfg.freeze()
    return trainer_lib, cfg
 
def main(config):
    # Ensure directories are setup
    # We feed network_config via unparsed_args so that it is flexible to deal with different models
    trainer = trainer_lib.Trainer(config)
    trainer.set_train_mode(True)
    trainer.get_dataloader()
    trainer.init_weight()
    trainer.train()





if __name__ == '__main__':

    parser = default_argument_parser()
    args = parser.parse_args()

    trainer_lib, config = setup(args)
    main(config)






