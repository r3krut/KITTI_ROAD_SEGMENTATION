"""
    The Main module
"""

import cv2
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

import utils
import img_utils as imutils

from models import RekNetM1
from utils import count_params
from losses import BCEJaccardLoss
from road_dataset import RoadDataset
from metrics import validation

from transforms import (
    train_transformations,
    valid_tranformations,
)

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

#For reproducibility
torch.manual_seed(111)

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Argument parser for the main module. Main module represents train procedure.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the root dir.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of model.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the KITTI dataset which contains 'testing' and 'training' subdirs.")
    
    #optimizer options
    parser.add_argument("--optim", type=str, default="SGD", help="Type of optimizer: SGD or Adam")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rates for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optim.")

    #other options
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of examples per batch.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of loading workers.")
    parser.add_argument("--device-ids", type=str, default="0", help="ID of devices for multiple GPUs.")
    parser.add_argument("--alpha", type=float, default=0, help="Modulation factor for custom loss.")
    parser.add_argument("--status-every", type=int, default=1, help="Status every parameter.")
    
    args = parser.parse_args()

    #Console logger definition
    console_logger = logging.getLogger("console-logger")
    console_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    console_logger.addHandler(ch)

    console_logger.info(args)

    #Model definition
    model = RekNetM1(num_classes=1, bn_enable=True)

    #Move model to devices
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    cudnn.benchmark = True

    #Loss definition
    loss = BCEJaccardLoss(alpha=args.alpha)
    
    train_dataset = RoadDataset(dataset_path=Path(args.dataset_path), transforms=train_transformations())
    valid_dataset = RoadDataset(dataset_path=Path(args.dataset_path), transforms=valid_tranformations(), is_valid=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    console_logger.info("Train dataset length: {}".format(len(train_dataset)))
    console_logger.info("Validation dataset length: {}".format(len(valid_dataset)))

    #Optim definition
    if args.optim == "SGD":
        optim = SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
        console_logger.info("Uses the SGD optimizer with lr={0} and momentum={1}".format(args.lr, args.momentum))
    else:
        optim = Adam(params=model.parameters(), lr=args.lr)
        console_logger.info("Uses the Adam optimizer with lr={0}".format(args.lr))

    valid = validation

    utils.train_routine(
        console_logger=console_logger,
        root=args.root_dir,
        model_name=args.model_name,
        model=model,
        criterion=loss,
        optimizer=optim,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        n_epochs=args.n_epochs,
        status_every=args.status_every
    )

if __name__ == "__main__":
    main()