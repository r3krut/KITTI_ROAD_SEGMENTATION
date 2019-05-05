"""
    This module contains some auxilary functions
"""

import os
import torch
import torch.nn as nn

import copy
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

from tb_logger import Logger

def count_params(model: nn.Module) -> (int, int):
    """
        Calculates the total and trainable parameters in model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (total, trainable)


def to_gpu(x: torch.Tensor):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def save_model(model_path: str, model, best_jaccard, best_dice, epoch):
    torch.save({"best_jaccard": best_jaccard, "best_dice": best_dice, "epoch": epoch, "model": model}, model_path)


def make_info_string(sep=',', **kwargs):
    """
        Construct an information string in the following view: key1: value1[sep]key2: value2[sep][(keyN: valueN)]
        params:
            sep         : a separator between instances. Possible values: ',', '\n'
            **kwargs    : params
    """
    
    if sep not in [',', '\n']:
        ValueError("Wrong separator: {}. 'sep' must be: ',' or '\n'".format(sep))
    
    info_str = ""
    for key, value in kwargs.items():
        info_str += "{0}: {1}{2} ".format(key, value, sep)
    info_str = info_str[:-2] if info_str[-2] == ',' else info_str[:-1]
    return info_str


def trainval_split(images_paths: str, masks_paths: str, fold=5):
    """
        Splits images and masks by two sets: train and validation by folds with a small stratification by categories 'uu', 'um' and 'umm'. 
        Possible value for 'fold' is: 1, 2, 3, 4, 5
        
        params:
            images_paths      :   dir with source images(without validation area)
            masks_paths       :   dir with masks
            fold              :   number of validation fold
    """

    images_paths = Path(images_paths)
    masks_paths = Path(masks_paths)

    images_paths = sorted(list(map(str, images_paths.glob("*"))))
    masks_paths = sorted(list(map(str, masks_paths.glob("*"))))

    if len(images_paths) < 5:
        raise RuntimeError("Length of images_paths less then 5.")

    if fold not in range(1,6):
        raise ValueError("Invalid fold number: {}. 'fold' can be 1,2,3,4 or 5.".format(fold))

    #Urban unmarked
    uu_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("uu_"), images_paths))
    uu_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("uu_"), masks_paths))

    #Urban marked
    um_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("um_"), images_paths))
    um_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("um_"), masks_paths))

    #Urban multiple mark
    umm_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("umm_"), images_paths))
    umm_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("umm_"), masks_paths))

    assert len(uu_imgs_paths) == len(uu_masks_paths), "Error. uu_imgs_paths and uu_masks_paths has differnet length."
    assert len(um_imgs_paths) == len(um_masks_paths), "Error. um_imgs_paths and um_masks_paths has differnet length."
    assert len(umm_imgs_paths) == len(umm_masks_paths), "Error. umm_imgs_paths and umm_masks_paths has differnet length."

    uu_imgs_per_fold = round(len(uu_imgs_paths) / 5)
    um_imgs_per_fold = round(len(um_imgs_paths) / 5)
    umm_imgs_per_fold = round(len(umm_imgs_paths) / 5)

    #train urban unmarked
    if fold == 5:
        #UU
        valid_uu_imgs_paths = uu_imgs_paths[-(len(uu_imgs_paths)-uu_imgs_per_fold*4):]
        valid_uu_masks_paths = uu_masks_paths[-(len(uu_imgs_paths)-uu_imgs_per_fold*4):]
        train_uu_imgs_paths = list(set(uu_imgs_paths) - set(valid_uu_imgs_paths))
        train_uu_masks_paths = list(set(uu_masks_paths) - set(valid_uu_masks_paths))

        #UM
        valid_um_imgs_paths = um_imgs_paths[-(len(um_imgs_paths)-um_imgs_per_fold*4):]
        valid_um_masks_paths = um_masks_paths[-(len(um_imgs_paths)-um_imgs_per_fold*4):]
        train_um_imgs_paths = list(set(um_imgs_paths) - set(valid_um_imgs_paths))
        train_um_masks_paths = list(set(um_masks_paths) - set(valid_um_masks_paths))

        #UMM
        valid_umm_imgs_paths = umm_imgs_paths[-(len(umm_imgs_paths)-umm_imgs_per_fold*4):]
        valid_umm_masks_paths = umm_masks_paths[-(len(umm_imgs_paths)-umm_imgs_per_fold*4):]
        train_umm_imgs_paths = list(set(umm_imgs_paths) - set(valid_umm_imgs_paths))
        train_umm_masks_paths = list(set(umm_masks_paths) - set(valid_umm_masks_paths))
    else:
        #UU
        valid_uu_imgs_paths = uu_imgs_paths[:fold*uu_imgs_per_fold][-uu_imgs_per_fold:]
        valid_uu_masks_paths = uu_masks_paths[:fold*uu_imgs_per_fold][-uu_imgs_per_fold:]
        train_uu_imgs_paths = list(set(uu_imgs_paths) - set(valid_uu_imgs_paths))
        train_uu_masks_paths = list(set(uu_masks_paths) - set(valid_uu_masks_paths))

        #UM
        valid_um_imgs_paths = um_imgs_paths[:fold*um_imgs_per_fold][-um_imgs_per_fold:]
        valid_um_masks_paths = um_masks_paths[:fold*um_imgs_per_fold][-um_imgs_per_fold:]
        train_um_imgs_paths = list(set(um_imgs_paths) - set(valid_um_imgs_paths))
        train_um_masks_paths = list(set(um_masks_paths) - set(valid_um_masks_paths))

        #UMM
        valid_umm_imgs_paths = umm_imgs_paths[:fold*umm_imgs_per_fold][-umm_imgs_per_fold:]
        valid_umm_masks_paths = umm_masks_paths[:fold*umm_imgs_per_fold][-umm_imgs_per_fold:]
        train_umm_imgs_paths = list(set(umm_imgs_paths) - set(valid_umm_imgs_paths))
        train_umm_masks_paths = list(set(umm_masks_paths) - set(valid_umm_masks_paths))

    #total train
    train_imgs_total = train_uu_imgs_paths + train_um_imgs_paths + train_umm_imgs_paths
    train_masks_total = train_uu_masks_paths + train_um_masks_paths + train_umm_masks_paths

    #total valid
    valid_imgs_total = valid_uu_imgs_paths + valid_um_imgs_paths + valid_umm_imgs_paths
    valid_masks_total = valid_uu_masks_paths + valid_um_masks_paths + valid_umm_masks_paths

    return ((train_imgs_total, train_masks_total), (valid_imgs_total, valid_masks_total))


def train_routine(console_logger: logging.Logger,
    root: str, 
    model_name: str,
    model: nn.Module, 
    criterion, 
    optimizer,
    scheduler,
    train_loader, 
    valid_loader,
    validation,
    n_epochs=100,
    status_every=5):

    """
        General trainig routine.
        params:
            console_logger          : logger object for logging
            model_name              : name of a training model
            model                   : model for training
            criterion               : loss function
            optimizer               : SGD, Adam or other
            scheduler               : learning rate scheduler
            train_loader            :
            valid_loader            :
            validation              : validation routine
            n_epochs                : number of training epochs
            status_every            : the parameter which controls the frequency of status printing
    """

    #Load model if it exists
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    model_root = root / model_name
    model_root.mkdir(exist_ok=True, parents=True)

    model_path = model_root / 'model.pt'
    logging_path = model_root / 'train.log'
    
    #file logger definition
    file_logger = logging.getLogger("file-logger")
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(logging_path), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    file_logger.addHandler(fh)

    #Logging to the TensorBoardX
    tbx_logger = Logger(log_dir=str(model_root / "tbxlogs"))

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state["epoch"]
        best_jaccard = state["best_jaccard"]
        best_dice = state["best_dice"]
        model.load_state_dict(state["model"]) 
        console_logger.info("\nModel '{0}' was restored. Best Jaccard: {1}, Best DICE: {2}, Epoch: {3}".format(str(model_path), best_jaccard, best_dice, epoch))
    else:
        epoch = 0
        best_jaccard = 0
        best_dice = 0
    
    epoch += 1
    n_epochs = n_epochs + epoch
    best_model = copy.deepcopy(model.state_dict())

    train_losses = []
    valid_losses = []
    jaccards = []
    dices = []

    for epoch in range(epoch, n_epochs):
        
        epoch_train_losses = []

        #Train mode
        model.train()
        
        #LR step for MultiStepLR scheduler
        scheduler.step()

        try:
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = to_gpu(inputs)
                targets = to_gpu(targets)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(targets, outputs)
                    loss.backward()
                    optimizer.step()
                epoch_train_losses.append(loss.item())

            #Train loss per epoch
            epoch_train_loss = np.mean(epoch_train_losses).astype(dtype=np.float64)
            
            #Validation
            valid_dict = validation(model, criterion, valid_loader)

            #LR step for ReduceOnPlateau
            # scheduler.step(valid_dict["val_jacc"])

            train_losses.append(epoch_train_loss)
            valid_losses.append(valid_dict["val_loss"])
            jaccards.append(valid_dict["val_jacc"])
            dices.append(valid_dict["val_dice"])

            if valid_dict["val_jacc"] > best_jaccard:
                best_jaccard = valid_dict["val_jacc"]
                best_dice = valid_dict["val_dice"]
                best_model = copy.deepcopy(model.state_dict())

            if epoch and (epoch % status_every == 0):
                info_str = "\nEpoch: {}\n".format(epoch)
                info_str += "-"*30
                info_str += "\nTrain loss: {0}".format(epoch_train_loss)
                info_str += "\nValid loss: {0}".format(valid_dict["val_loss"]) 
                info_str += "\nValid Jaccard: {0}".format(valid_dict["val_jacc"]) 
                info_str += "\nValid DICE: {0}\n".format(valid_dict["val_dice"]) 
                info_str += "-"*30
                info_str += "\n"
                console_logger.info(info_str)
                
                #Log to file
                info_str = "\nepoch: {}, ".format(epoch)
                info_str += "train_loss: {}, ".format(epoch_train_loss)
                info_str += "val_loss: {}, ".format(valid_dict["val_loss"])
                info_str += "val_jaccard: {}, ".format(valid_dict["val_jacc"])
                info_str += "val_dice: {}\n".format(valid_dict["val_dice"])
                file_logger.info(info_str)

                #Log to the tbX
                tbx_logger.log_scalars(tag="losses", values={"train_loss": epoch_train_loss, "valid_loss": valid_dict["val_loss"]}, step=epoch)
                tbx_logger.log_scalars(tag="metrics", values={"jaccard": valid_dict["val_jacc"], "DICE": valid_dict["val_dice"]}, step=epoch)

        except KeyboardInterrupt:
            console_logger.info("KeyboardInterrupt, saving snapshot.")
            save_model(str(model_path), best_model, best_jaccard, best_dice, epoch)
            console_logger.info("Done!")
    
    info_str = "\nTraining process is done!\n" + "*"*30
    info_str += "\nTrain loss: {0}".format(np.mean(train_losses).astype(dtype=np.float64)) 
    info_str += "\nValid loss: {0}".format(np.mean(valid_losses).astype(dtype=np.float64))
    info_str += "\nMean Jaccard: {0}".format(np.mean(jaccards).astype(dtype=np.float64))
    info_str += "\nMean DICE: {0}".format(np.mean(dices).astype(dtype=np.float64))
    info_str += "\nBest Jaccard: {0}".format(best_jaccard)
    info_str += "\nBest DICE: {0}\n".format(best_dice) + "*"*30

    console_logger.info(info_str)
    file_logger.info(info_str)

    #model saving
    save_model(str(model_path), best_model, best_jaccard, best_dice, n_epochs+1)


def shutdown_system():
    os.system("shutdown -h now")