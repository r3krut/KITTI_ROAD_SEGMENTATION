"""
    This module contains the code which performs evaluation of model(models) on hold-out dataset.
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

from utils.utils import to_gpu, fmeasure_evaluation

from models.reknetm1 import RekNetM1
from models.reknetm2 import RekNetM2
from models.lidcamnet_fcn import LidCamNet

from .metrics import (
    jaccard,
    dice,
    evalExp,
    pxEval_maximizeFMeasure
)

#Dataset 
from data_processing.road_dataset import RoadDataset2

from .transforms import valid_tranformations

#torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(path2models, model: nn.Module, threshold: float, holdout_dataset: str, evaluate_one=True):
    """
        Perform evaluation of a model or list of models.
        
        Params:
            path2models     : path to a model(or models in cross-validation case)
            model           : A model. Can be RekNetM1, etc
            threshold       : 
            holdout_datset  : path to a hold-out dataset(must contains 'imgs' and 'masks' subdirs)
            evaluate_one    : if True then perform evaluation of one model. Otherwise evaluation of multiple models(in case of cross-val)
    """

    assert threshold >= 0 and threshold < 1.0, "Error. Invalid threshold: {}".format(threshold)

    path2models = Path(path2models)
    holdout_dataset = Path(holdout_dataset)

    img_paths = list(map(str, (holdout_dataset / 'imgs').glob('*')))
    mask_paths = list(map(str, (holdout_dataset / 'masks').glob('*')))

    test_dataset = RoadDataset2(img_paths=img_paths, mask_paths=mask_paths, transforms=valid_tranformations())
    fmax_test_datset = RoadDataset2(img_paths=img_paths, mask_paths=mask_paths, transforms=valid_tranformations(), fmeasure_eval=True)

    #metrics lists
    jaccards = []
    dices = []

    if evaluate_one:
        path2models = path2models / 'model.pt'
        
        print("Evluation for the single model: {}".format(str(path2models)))

        if not path2models.exists():
            raise RuntimeError("Model {} does not exists.".format(str(path2models)))
        
        state = torch.load(str(path2models))
        model.load_state_dict(state["model"])
        
        #eval mode
        model.eval()
        
        for idx, data in enumerate(test_dataset):
            img, mask = data
            img = to_gpu(img.unsqueeze(0).contiguous())
            mask = to_gpu(mask.unsqueeze(0).contiguous())

            with torch.set_grad_enabled(False):
                predict = model(img)

            predict = F.sigmoid(predict)
            jacc = jaccard(mask, (predict > threshold).float()) 
            d = dice(mask, (predict > threshold).float())
            
            jaccards.append(jacc)
            dices.append(d)

        evaluation_jaccard = np.mean(jaccards).astype(dtype=np.float64)
        evaluation_dice = np.mean(dices).astype(dtype=np.float64)
        uu_metrics, um_metrics, umm_metrics = fmeasure_evaluation([model], valid_dataset=fmax_test_datset)

        return {"eval_jacc" : evaluation_jaccard, "eval_dice" : evaluation_dice}, uu_metrics, um_metrics, umm_metrics
    else:
        #Imporant! path2models dir should contains a few subdirs. These subdirs by itself contains models which were trained on folds.
        list_models_paths = sorted(list(path2models.glob('*')))

        print("Evaluation for multiple models: {}".format([str(lmp/'model.pt') for lmp in list_models_paths]))

        models_list = []
        for lmp in list_models_paths:
            model_path = lmp / 'model.pt'

            if not model_path.exists():
                raise RuntimeError("Model {} does not exists.".format(str(model_path)))

            state = torch.load(str(model_path))
            model.load_state_dict(state["model"])

            models_list.append(model.eval())

        #Evaluate on the test data
        for idx, data in enumerate(test_dataset):
            img, mask = data
            img = to_gpu(img.unsqueeze(0).contiguous())
            mask = to_gpu(mask.unsqueeze(0).contiguous())

            #Averaging all predictions for one point of test data
            sum_predicts = to_gpu(torch.zeros(mask.shape).float())
            for m in models_list:
                with torch.set_grad_enabled(False):
                    predict = m(img)
                sum_predicts += F.sigmoid(predict)

            predict = (sum_predicts / len(models_list)).float()

            jacc = jaccard(mask, (predict > threshold).float()) 
            d = dice(mask, (predict > threshold).float())
            
            jaccards.append(jacc)
            dices.append(d)

        evaluation_jaccard = np.mean(jaccards).astype(dtype=np.float64)
        evaluation_dice = np.mean(dices).astype(dtype=np.float64)
        uu_metrics, um_metrics, umm_metrics = fmeasure_evaluation(models_list, valid_dataset=fmax_test_datset)

        return {"eval_jacc" : evaluation_jaccard, "eval_dice" : evaluation_dice}, uu_metrics, um_metrics, umm_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation module params.")
    parser.add_argument("--models-paths", type=str, required=True, help="Path to a single model(or multiple models)")
    parser.add_argument("--model-type", type=str, default="reknetm1", help="Type of model. Can be reknetm1 or reknetm2.")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--holdout-path", type=str, required=True, help="Path to a dir which contains hold-out dataset for evaluation(must contains 'imgs' and 'masks' subdirs).")
    parser.add_argument("--eval-one", type=int, default=1)

    args = parser.parse_args()

    if args.model_type == "reknetm1":
        model = RekNetM1(num_classes=1, 
                ebn_enable=True, 
                dbn_enable=True, 
                upsample_enable=False, 
                act_type="celu",
                init_type="He")
    elif args.model_type == "reknetm2":
        model = RekNetM2(num_classes=1, 
                ebn_enable=True, 
                dbn_enable=True, 
                upsample_enable=False, 
                act_type="celu",
                init_type="He",
                attention=True)
    elif args.model_type == "lcn":
        model = LidCamNet(num_classes=1,
                bn_enable=False)
    else:
        raise ValueError("Unknown model: {}".format(args.model_type))

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=None).cuda()

    jd_metrics, uu_metrics, um_metrics, umm_metrics = evaluate(path2models=args.models_paths, model=model, threshold=args.thresh, holdout_dataset=args.holdout_path, evaluate_one=bool(args.eval_one))

    print("Evaluation done for {} model(s) {}!".format(("single" if bool(args.eval_one) else "multiple"), args.models_paths))
    print("Evaluation Jaccard: {}".format(jd_metrics["eval_jacc"]))
    print("Evaluation DICE: {}\n".format(jd_metrics["eval_dice"]))

    print("UU_MaxF: {}, UU_AvgPrec: {}, UU_PRE: {}, UU_REC: {}".format(uu_metrics["MaxF"], uu_metrics["AvgPrec"], uu_metrics["PRE_wp"][0], uu_metrics["REC_wp"][0]))
    print("UM_MaxF: {}, UM_AvgPrec: {}, UM_PRE: {}, UM_REC: {}".format(um_metrics["MaxF"], um_metrics["AvgPrec"], um_metrics["PRE_wp"][0], um_metrics["REC_wp"][0]))
    print("UMM_MaxF: {}, UMM_AvgPrec: {}, UMM_PRE: {}, UMM_REC: {}".format(umm_metrics["MaxF"], umm_metrics["AvgPrec"], umm_metrics["PRE_wp"][0], umm_metrics["REC_wp"][0]))