import argparse
import yaml
import sys
import os 
import torch
import numpy as np
import torch.utils.tensorboard as tb
import logging
import shutil
import traceback
import importlib
import random


torch.set_printoptions(sci_mode = False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--alg", 
        type=str, 
        required=True, 
        help="Choose an algorithm to run"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="whether to train the model"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to test the model"
    )
    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical"
    )

    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="A string for experiment comment"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed"
    )

    args = parser.parse_args()
    args.alg_path = os.path.join("algorithms", args.alg)
    args.log_path = os.path.join(args.alg_path, "logs")
    tb_path = os.path.join(args.alg_path, "tensorboard")

    # parse config file
    with open(os.path.join(args.alg_path, "config.yml"), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)

    if args.train:
        if os.path.exists(args.log_path):
            shutil.rmtree(args.log_path)
            shutil.rmtree(tb_path)
        os.makedirs(args.log_path)
        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        handler2.setFormatter(formatter)
    new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    if args.train:
        logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device:{}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value) 
    return namespace

def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    
    #params_agent = importlib.import_module(args.alg_path+".agent")
    #agent = params_agent.Agent()
    params = importlib.import_module("algorithms."+args.alg +".main")
    runner = params.Runner(args, config)
    try:
        if args.train:
            runner.train()
        elif args.test:
            runner.test()
    except Exception:
        logging.info(traceback.format_exc())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())