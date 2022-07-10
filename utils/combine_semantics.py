#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script combines moving object segmentation with semantic information
# example: the class 'car' is further subdivided into 'static car' and 'moving car'

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

from utils_Kitti360 import load_files, load_labels
#from utils import load_files, load_labels

def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int16)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int16)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

if __name__ == '__main__':
  # load config file
  config_filename = 'config/post-processing.yaml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))

  # specify inference dir
  inference_root = config['inference_root']

  # specify moving object segmentation results folder
  mos_pred_root = os.path.join(inference_root, 'SalsaNext_mos')
  
  # specify semantic segmentation results folder
  semantic_pred_root = os.path.join(inference_root, 'SalsaNext_semantics')
  
  # create a new folder for combined results
  combined_results_root = os.path.join(inference_root, 'SalsaNext_combined_semantics_mos')
  
  # specify the split
  split = config['split']
  data_yaml = yaml.load(open('config/combine_mos_semantics.yaml'))
  combine_mos_semantic_map = data_yaml['combine_mos_semantic_map']
  
  # create output folder
  seqs = []
  if not os.path.exists(os.path.join(combined_results_root)):
    os.makedirs(os.path.join(combined_results_root))
  
  if split == 'train':
    print(data_yaml["split"]["train"])
    for seq in data_yaml["split"]["train"]:
      #seq = '2013_05_28_drive_%04d_sync' %seq #KITTI-360
      seq = '{0:02d}'.format(int(seq))  # KITTI odometry
      print("train", seq)
      if not os.path.exists(os.path.join(combined_results_root, seq)):
        os.makedirs(os.path.join(combined_results_root, seq))
      if not os.path.exists(os.path.join(combined_results_root, seq, "predictions")):
        os.makedirs(os.path.join(combined_results_root, seq, "predictions"))
      seqs.append(seq)
  if split == 'valid':
    for seq in data_yaml["split"]["valid"]:
      #seq = '2013_05_28_drive_%04d_sync' %seq #KITTI-360
      seq = '{0:02d}'.format(int(seq)) # KITTI odometry
      print("valid", seq)
      if not os.path.exists(os.path.join(combined_results_root, seq)):
        os.makedirs(os.path.join(combined_results_root, seq))
      if not os.path.exists(os.path.join(combined_results_root, seq, "predictions")):
        os.makedirs(os.path.join(combined_results_root, seq, "predictions"))
      seqs.append(seq)
  if split == 'test':
    for seq in data_yaml["split"]["test"]:
      #seq = '2013_05_28_drive_%04d_sync' %seq #KITTI-360
      seq = '{0:02d}'.format(int(seq)) # KITTI odometry
      print("test", seq)
      if not os.path.exists(os.path.join(combined_results_root, seq)):
        os.makedirs(os.path.join(combined_results_root, seq))
      if not os.path.exists(os.path.join(combined_results_root, seq, "predictions")):
        os.makedirs(os.path.join(combined_results_root, seq, "predictions"))
      seqs.append(seq)
  
  for seq in seqs:
    # load moving object segmentation files
    mos_pred_seq_path = os.path.join(mos_pred_root, seq, "predictions")
    mos_pred_files = load_files(mos_pred_seq_path)
    
    # load semantic segmentation files
    semantic_pred_seq_path = os.path.join(semantic_pred_root, seq, "predictions")
    semantic_pred_files = load_files(semantic_pred_seq_path)
    
    print('processing seq:', seq)

    assert(len(mos_pred_files) == len(semantic_pred_files))

    movable_classes = []
    for key in combine_mos_semantic_map.keys():
      movable_classes.append(key)

    for frame_idx in tqdm(range(len(mos_pred_files))):
      mos_pred = load_labels(mos_pred_files[frame_idx])  # mos_pred should be 1 or 2 for static/dynamic
      semantic_pred = load_labels(semantic_pred_files[frame_idx])

      assert(len(mos_pred) == len(semantic_pred))

      combine_pred = semantic_pred

      for sem, mos, com in zip(semantic_pred, mos_pred, range(len(combine_pred))):
        if sem in movable_classes:
          if mos == 2:
            label_dyn = map(sem, combine_mos_semantic_map)
            combine_pred[com] = label_dyn

      assert(semantic_pred_files[frame_idx].split('/')[-1].split('.')[0] == mos_pred_files[frame_idx].split('/')[-1].split('.')[0])
  
      file_name = os.path.join(combined_results_root, seq, 'predictions', semantic_pred_files[frame_idx].split('/')[-1].split('.')[0])
      combine_pred.reshape((-1)).astype(np.int16)
      combine_pred.tofile(file_name + '.bin')