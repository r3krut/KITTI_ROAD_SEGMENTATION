"""
    This module contains some auxilary functions
"""

import torch
import torch.nn as nn

import copy
import json
from pathlib import Path
from datetime import datetime

import numpy as np

def count_params(model: nn.Module) -> (int, int):
    """
        Calculates the total and trainable parameters in model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (total, trainable)


def to_gpu(x: torch.Tensor):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def save_model(model_path: str, model: nn.Module, best_jaccard, best_dice, epoch):
    torch.save({"best_jaccard": best_jaccard, "best_dice": best_dice, "epoch": epoch, "model": model.state_dict}, model_path)


def logging(file_path, epoch: int, **data):
    """
        Preforms logging to file
    """
    data['epoch'] = epoch
    data['dt'] = datetime.now().isoformat()
    file_path.write(json.dumps(data, sort_keys=True))
    file_path.write('\n')
    file_path.flush()


def train_routine(root: str, 
    model_name: str,
    model: nn.Module, 
    criterion, 
    optimizer,
    train_loader, 
    valid_loader,
    validation,
    n_epochs=100,
    status_every=5):

    """
        General trainig routine.
        params:
            model_name              : name of a training model
            model                   : model for training
            criterion               : loss function
            optimizer               : SGD, Adam or other
            train_loader            :
            valid_loader            :
            validation              : validation routine
            n_epochs                : number of training epochs
            status_every            : the parameter which controls the frequency of status printing
    """

    #Load model if it exists
    root = Path(root)
    model_root = root / 'models' / model_name
    model_path = model_root / 'model.pt'
    logging_path = model_root / 'logging.log'

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state["epoch"]
        best_jaccard = state["best_jaccard"]
        best_dice = state["best_dice"]
        model.load_state_dict(state["model"])
        print("Model '{0}' was restored. Best Jaccard: {1}, Best DICE: {2}, Epoch: {3}".format(str(model_path), best_jaccard, best_dice, epoch)) 
    else:
        epoch = 0
    epoch += 1

    best_jaccard = 0
    best_dice = 0
    best_model = copy.deepcopy(model.state_dict())

    train_losses = []
    valid_losses = []
    jaccards = []
    dices = []

    for epoch in range(epoch, n_epochs + 1):
        
        epoch_train_losses = []

        #Train mode
        model.train()
        try:
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = to_gpu(inputs)
                targets = to_gpu(targets)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                epoch_train_losses.append(loss.item())

            #Train loss per epoch
            epoch_train_loss = np.mean(epoch_train_losses).astype(dtype=np.float64)
            
            #Validation
            valid_dict = validation(model, criterion, valid_loader)
            
            train_losses.append(epoch_train_loss)
            valid_losses.append(valid_dict["val_loss"])
            jaccards.append(valid_dict["val_jacc"])
            dices.append(valid_dict["val_dice"])

            if valid_dict["val_jacc"] > best_jaccard:
                best_jaccard = valid_dict["val_jacc"]
                best_dice = valid_dict["val_dice"]
                best_model = copy.deepcopy(model.state_dict())

            if epoch and (epoch % status_every == 0):
                print("Epoch: {}".format(epoch))
                print("-"*30)
                print("Train loss: {0}".format(epoch_train_loss))
                print("Valid loss: {0}".format(valid_dict["val_loss"]))
                print("Valid Jaccard: {0}".format(valid_dict["val_jacc"]))
                print("Valid DICE: {0}".format(valid_dict["val_dice"]))
                print("-"*30)
                print()
                valid_dict["train_loss"] = epoch_train_loss
                #Log to file
                logging(logging_path, epoch, **valid_dict)

        except KeyboardInterrupt:
            print("KeyboardInterrupt, saving snapshot.")
            save_model(model_path, best_model, best_jaccard, best_dice, epoch)
            print("Done!")
    
    print("\nTraining process is done!")
    print("*"*30)
    print("Train loss: {0}".format(np.mean(train_losses).astype(dtype=np.float64)))
    print("Valid loss: {0}".format(np.mean(valid_losses).astype(dtype=np.float64)))
    print("Mean Jaccard: {0}".format(np.mean(jaccards).astype(dtype=np.float64)))
    print("Mean DICE: {0}".format(np.mean(dices).astype(dtype=np.float64)))
    print("Best Jaccard: {0}".format(best_jaccard))
    print("Beast DICE: {0}".format(best_dice))
    print("*"*30)

    #model saving
    save_model(model_path, best_model, best_jaccard, best_dice, n_epochs+1)
    
    
    
    