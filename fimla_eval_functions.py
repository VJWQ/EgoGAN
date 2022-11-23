import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import torch
import numpy as np
import cv2
"""functions for aggregating scores"""

import torch.nn as nn
from torchvision.utils import save_image


def _compare_maps(gt, pred, all_thresh):
  """ compare two maps """
  # return tp, tn, fp, fn
  N, Z = 1, gt.shape[1] 
  gt_map, pred_map = gt, pred
  # not technicallly correct, yet should be very similar to the true PR
  tp = np.zeros((all_thresh.shape[0],))
  fp = np.zeros((all_thresh.shape[0],))
  fn = np.zeros((all_thresh.shape[0],))
  tn = np.zeros((all_thresh.shape[0],))

  # # get valid slices
  valid_slice = []
  for slice_idx in range(N):
    if np.max(gt_map[slice_idx, :, :]) > 0.01:
      valid_slice.append(slice_idx)
  # reslice the data
  valid_gt = gt_map[valid_slice, :, :]
  valid_gt = (valid_gt>0.01)
  valid_pred = pred_map[valid_slice, :, :]

  for idx, thresh in enumerate(all_thresh):
    mask = (valid_pred>=thresh)
    tp[idx] += np.sum(np.logical_and(mask==1, valid_gt==1))
    tn[idx] += np.sum(np.logical_and(mask==0, valid_gt==0))
    fp[idx] += np.sum(np.logical_and(mask==1, valid_gt==0))
    fn[idx] += np.sum(np.logical_and(mask==0, valid_gt==1))

  return tp, tn, fp, fn

def main():

  output_file = ""

  """Evaluation of action localization"""
  with open(output_file, 'rb') as f:
    preds_list,labels_list, videoid_list = pickle.load(f)

  all_preds = torch.cat(preds_list)
  all_labels = torch.cat(labels_list)

  # # tp / fp / fn / tn 
  all_thresh = np.linspace(0, 1.0, 41)
  f_tp = np.zeros((all_thresh.shape[0],))
  f_fp = np.zeros((all_thresh.shape[0],))
  f_fn = np.zeros((all_thresh.shape[0],))
  f_tn = np.zeros((all_thresh.shape[0],))

  l_tp = np.zeros((all_thresh.shape[0],))
  l_fp = np.zeros((all_thresh.shape[0],))
  l_fn = np.zeros((all_thresh.shape[0],))
  l_tn = np.zeros((all_thresh.shape[0],))

  m_tp = np.zeros((all_thresh.shape[0],))
  m_fp = np.zeros((all_thresh.shape[0],))
  m_fn = np.zeros((all_thresh.shape[0],))
  m_tn = np.zeros((all_thresh.shape[0],))

  print(len(all_labels))

  for idx in range(len(all_labels)):
    f_pred = torch.cat((torch.sigmoid(all_preds[idx][0][0]).unsqueeze(0), torch.sigmoid(all_preds[idx][1][0]).unsqueeze(0)), dim=0).unsqueeze(1)
    f_label = torch.cat((all_labels[idx][0][0].unsqueeze(0), all_labels[idx][1][0].unsqueeze(0)), dim=0).unsqueeze(1)

    m_pred = torch.cat((torch.sigmoid(all_preds[idx][0][1]).unsqueeze(0), torch.sigmoid(all_preds[idx][1][1]).unsqueeze(0)), dim=0).unsqueeze(1)
    m_label = torch.cat((all_labels[idx][0][1].unsqueeze(0), all_labels[idx][1][1].unsqueeze(0)), dim=0).unsqueeze(1)

    l_pred = torch.cat((torch.sigmoid(all_preds[idx][0][2]).unsqueeze(0), torch.sigmoid(all_preds[idx][1][2]).unsqueeze(0)), dim=0).unsqueeze(1)
    l_label = torch.cat((all_labels[idx][0][2].unsqueeze(0), all_labels[idx][1][2].unsqueeze(0)), dim=0).unsqueeze(1)

    f_ctp, f_ctn, f_cfp, f_cfn = _compare_maps(
    np.squeeze(f_label.numpy()), np.squeeze(f_pred.numpy()), all_thresh)
    f_tp = f_tp + f_ctp
    f_tn = f_tn + f_ctn
    f_fp = f_fp + f_cfp
    f_fn = f_fn + f_cfn

    m_ctp, m_ctn, m_cfp, m_cfn = _compare_maps(
    np.squeeze(m_label.numpy()), np.squeeze(m_pred.numpy()), all_thresh)
    m_tp = m_tp + m_ctp
    m_tn = m_tn + m_ctn
    m_fp = m_fp + m_cfp
    m_fn = m_fn + m_cfn

    l_ctp, l_ctn, l_cfp, l_cfn = _compare_maps(
    np.squeeze(l_label.numpy()), np.squeeze(l_pred.numpy()), all_thresh)
    l_tp = l_tp + l_ctp
    l_tn = l_tn + l_ctn
    l_fp = l_fp + l_cfp
    l_fn = l_fn + l_cfn

  # prec / recall
  f_prec = f_tp / (f_tp+f_fp+1e-6)
  f_recall = f_tp / (f_tp+f_fn+1e-6)
  f_f1 = 2*f_prec*f_recall / (f_prec + f_recall + 1e-6)
  f_idx = np.argmax(f_f1)
  print("F1 Score of short-term {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
    f_f1[f_idx], f_prec[f_idx], f_recall[f_idx], all_thresh[f_idx]))
  print("****************************************")
  m_prec = m_tp / (m_tp+m_fp+1e-6)
  m_recall = m_tp / (m_tp+m_fn+1e-6)
  m_f1 = 2*m_prec*m_recall / (m_prec + m_recall + 1e-6)
  m_idx = np.argmax(m_f1)
  print("F1 Score of middle-term {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
    m_f1[m_idx], m_prec[m_idx], m_recall[m_idx], all_thresh[m_idx]))
  print("****************************************")
  l_prec = l_tp / (l_tp+l_fp+1e-6)
  l_recall = l_tp / (l_tp+l_fn+1e-6)
  l_f1 = 2*l_prec*l_recall / (l_prec + l_recall + 1e-6)
  l_idx = np.argmax(l_f1)
  print("F1 Score of long-term {:0.4f} (P={:0.4f}, R={:0.4f}) at th={:0.4f}".format(
    l_f1[l_idx], l_prec[l_idx], l_recall[l_idx], all_thresh[l_idx]))


if __name__ == "__main__":
    main()
