"""
    This module contains some metrics
"""

import torch
import numpy as np

def batch_jaccard(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + predict.sum(dim=-2).sum(dim=-1)
    jaccard = (intersection + eps) / (union - intersection + eps)
    return list(jaccard.data.cpu().numpy())


def batch_dice(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + predict.sum(dim=-2).sum(dim=-1)
    dice = (2 * intersection + eps) / (union + eps)
    return list(dice.data.cpu().numpy())


def jaccard(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum().float()
    union = target.sum().float() + predict.sum().float()
    return ((intersection + eps) / (union - intersection + eps)).cpu().item()


def dice(target: torch.Tensor, predict: torch.Tensor):
    eps = 1e-15
    intersection = (target * predict).sum()
    union = target.sum() + predict.sum()
    return ((2 * intersection + eps) / (union + eps)).cpu().item()


#This code was taken from https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
def calculate_confusion_matrix_from_arrays(predictions, targets, nr_labels):
    replace_indices = np.vstack((
        targets.flatten(),
        predictions.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(dtype=np.uint32)
    return confusion_matrix


#This code was taken from https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
def calculate_jaccards(confusion_matrix):
    jaccards = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            jaccard = 0
        else:
            jaccard = float(true_positives) / denom
        jaccards.append(jaccard)
    return jaccards


#This code was taken from https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
def calculate_dices(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices


#This default code for evaluation in KITTI benchmark
def evalExp(gtBin, cur_prob, thres, validMap = None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_prob:
    :param thres:
    :param validMap:
    '''

    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
    
    #Merge validMap with validArea
    if not validMap is None:
        if not validArea is None:
            validMap = (validMap == True) & (validArea == True)
    elif not validArea is None:
        validMap = validArea

    # histogram of false negatives
    if not validMap is None:
        fnArray = cur_prob[(gtBin == True) & (validMap == True)]
    else:
        fnArray = cur_prob[(gtBin == True)]
        
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)]
    
    if not validMap is None:
        fpArray = cur_prob[(gtBin == False) & (validMap == True)]
    else:
        fpArray = cur_prob[(gtBin == False)]
    
    fpHist = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    if not validMap is None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum


#This default code for evaluation in KITTI benchmark
def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
    '''

    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''

    #Calc missing stuff
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP

    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    precision =  totalTP / (totalTP + totalFP + 1e-10)
    
    selector_invalid = (recall==0) & (precision==0)
    recall = recall[~selector_invalid]
    precision = precision[~selector_invalid]
        
    maxValidIndex = len(precision)
    
    #Pascal VOC average precision
    AvgPrec = 0
    counter = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        if ind is None:
            continue
        pmax = max(precision[ind])
        AvgPrec += pmax
        counter += 1
    AvgPrec = AvgPrec/counter
    
    # F-measure operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    MaxF = F[index]
    
    recall_bst = recall[index]
    precision_bst =  precision[index]

    TP = totalTP[index]
    TN = totalTN[index]
    FP = totalFP[index]
    FN = totalFN[index]
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    #ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
    prob_eval_scores = calcEvalMeasures(valuesMaxF)
    prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF

    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    prob_eval_scores['thresh'] = thresh
    if not thresh is None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh

    #return a dict
    return prob_eval_scores


#This default code for evaluation in KITTI benchmark
def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    correct_rate = A
    
    outDict = dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict