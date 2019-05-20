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

from models import RekNetM1, RekNetM2
from utils import count_params
from losses import BCEJaccardLoss, CCEJaccardLoss
from road_dataset import RoadDataset, RoadDataset2

from transforms import (
    train_transformations,
    valid_tranformations,
)

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from data_processing import (
    droped_valid_image_2_dir,
    train_masks_dir,
    crossval_split,
    image_2_dir
)

#For reproducibility
torch.manual_seed(111)

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Argument parser for the main module. Main module represents train procedure.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the root dir where will be stores models.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the KITTI dataset which contains 'testing' and 'training' subdirs.")
    parser.add_argument("--fold", type=int, default=1, help="Num of a validation fold.")
    
    #optimizer options
    parser.add_argument("--optim", type=str, default="SGD", help="Type of optimizer: SGD or Adam")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rates for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optim.")
    
    #Scheduler options
    parser.add_argument("--scheduler", type=str, default="multi-step", help="Type of a scheduler for LR scheduling.")
    parser.add_argument("--step-st", type=int, default=5, help="Step size for StepLR scheudle.")
    parser.add_argument("--milestones", type=str, default="30,70,90", help="List with milestones for MultiStepLR schedule.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma parameter for StepLR and MultiStepLR schedule.")
    parser.add_argument("--patience", type=int, default=5, help="Patience parameter for ReduceLROnPlateau schedule.")
    

    #model params
    parser.add_argument("--model-type", type=str, default="reknetm1", help="Type of model. Can be 'RekNetM1' or 'RekNetM2'.")
    parser.add_argument("--decoder-type", type=str, default="up", help="Type of decoder module. Can be 'up'(Upsample) or 'ConvTranspose2D'.")
    parser.add_argument("--init-type", type=str, default="He", help="Initialization type. Can be 'He' or 'Xavier'.")
    parser.add_argument("--act-type", type=str, default="relu", help="Activation type. Can be ReLU, CELU or FTSwish+.")
    parser.add_argument("--enc-bn-enable", type=int, default=1, help="Batch normalization enabling in encoder module.")
    parser.add_argument("--dec-bn-enable", type=int, default=1, help="Batch normalization enabling in decoder module.")
    # parser.add_argument("--skip-conn", type=int, default=0, help="UNet-like skip-connections enabling.")

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

    #number of classes
    num_classes = 1

    if args.decoder_type == "up":
        upsample_enable = True
        console_logger.info("Decoder type is Upsample.")
    elif args.decoder_type == "convTr":
        upsample_enable = False
        console_logger.info("Decoder type is ConvTranspose2D.")

    #Model definition
    if args.model_type == "reknetm1":
        model = RekNetM1(num_classes=num_classes, 
            ebn_enable=bool(args.enc_bn_enable),
            dbn_enable=bool(args.dec_bn_enable), 
            upsample_enable=upsample_enable, 
            act_type=args.act_type,
            init_type=args.init_type)
        console_logger.info("Uses RekNetM1 as the model.")
    elif args.model_type == "reknetm2":
        model = RekNetM2(num_classes=num_classes,
            ebn_enable=bool(args.enc_bn_enable), 
            dbn_enable=bool(args.dec_bn_enable), 
            upsample_enable=upsample_enable, 
            act_type=args.act_type,
            init_type=args.init_type)
        console_logger.info("Uses RekNetM2 as the model.")
    else:
        raise ValueError("Unknown model type: {}".format(args.model_type))

    console_logger.info("Number of trainable parameters: {}".format(utils.count_params(model)[1]))

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

    dataset_path = Path(args.dataset_path)
    images = str(dataset_path / "training" / droped_valid_image_2_dir)
    masks = str(dataset_path / "training" / train_masks_dir)

    #train-val splits for cross-validation by a fold
    ((train_imgs, train_masks), 
        (valid_imgs, valid_masks)) = crossval_split(images_paths=images, masks_paths=masks, fold=args.fold)

    train_dataset = RoadDataset2(img_paths=train_imgs, mask_paths=train_masks, transforms=train_transformations())
    valid_dataset = RoadDataset2(img_paths=valid_imgs, mask_paths=valid_masks, transforms=valid_tranformations())
    valid_fmeasure_datset = RoadDataset2(img_paths=valid_imgs, mask_paths=valid_masks, transforms=valid_tranformations(), fmeasure_eval=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=torch.cuda.device_count(), num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    console_logger.info("Train dataset length: {}".format(len(train_dataset)))
    console_logger.info("Validation dataset length: {}".format(len(valid_dataset)))

    #Optim definition
    if args.optim == "SGD":
        optim = SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
        console_logger.info("Uses the SGD optimizer with initial lr={0} and momentum={1}".format(args.lr, args.momentum))
    else:
        optim = Adam(params=model.parameters(), lr=args.lr)
        console_logger.info("Uses the Adam optimizer with initial lr={0}".format(args.lr))

    if args.scheduler == "step":
        lr_scheduler = StepLR(optimizer=optim, step_size=args.step_st, gamma=args.gamma)
        console_logger.info("Uses the StepLR scheduler with step={} and gamma={}.".format(args.step_st, args.gamma))
    elif args.scheduler == "multi-step":
        lr_scheduler = MultiStepLR(optimizer=optim, milestones=[int(m) for m in (args.milestones).split(",")], gamma=args.gamma)
        console_logger.info("Uses the MultiStepLR scheduler with milestones=[{}] and gamma={}.".format(args.milestones, args.gamma))
    elif args.scheduler == "rlr-plat":
        lr_scheduler = ReduceLROnPlateau(optimizer=optim, patience=args.patience, verbose=True)
        console_logger.info("Uses the ReduceLROnPlateau scheduler.")
    else:
        raise ValueError("Unknown type of schedule: {}".format(args.scheduler))

    valid = utils.binary_validation_routine

    utils.train_routine(
        args=args,
        console_logger=console_logger,
        root=args.root_dir,
        model=model,
        criterion=loss,
        optimizer=optim,
        scheduler=lr_scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        fm_eval_dataset=valid_fmeasure_datset,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes,
        n_epochs=args.n_epochs,
        status_every=args.status_every
    )

if __name__ == "__main__":
    main()