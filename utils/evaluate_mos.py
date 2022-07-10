#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by: Xieyuanli Chen

import argparse
import os
import yaml
import sys
import numpy as np
from auxiliary.np_ioueval import iouEval

# possible splits
splits = ["train", "valid", "test"]

def save_to_log(logdir,logfile,message):
    f = open(logdir+'/'+logfile, "a")
    f.write(message+'\n')
    f.close()
    return

# possible backends
backends = ["numpy", "torch"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_mos.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--backend', '-b',
      type=str,
      required=False,
      choices= ["numpy", "torch"],
      default="numpy",
      help='Backend for evaluation. One of ' +
      str(backends) + ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--config_labels', '-cl',
      type=str,
      required=False,
      default='config/semantic-kitti-mos.yaml',
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--config_pred', '-cp',
      type=str,
      required=False,
      default='config/eval_mos.yaml',
      help='Dataset config file. Defaults to %(default)s',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # fill in real predictions dir
  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Backend: ", FLAGS.backend)
  print("Split: ", FLAGS.split)
  print("Label config from: ", FLAGS.config_labels)
  print("Pred config from: ", FLAGS.config_pred)
  print("*" * 80)

  # assert split
  assert(FLAGS.split in splits)

  # assert backend
  assert(FLAGS.backend in backends)

  print("Opening data config file for labels from %s" % FLAGS.config_labels)
  DATA_LABELS = yaml.safe_load(open(FLAGS.config_labels, 'r'))
  print("Opening data config file for predicitons from %s" % FLAGS.config_pred)
  DATA_PRED = yaml.safe_load(open(FLAGS.config_pred, 'r'))

  # get number of interest classes, and the label mappings
  class_strings_lab = DATA_LABELS["labels"]
  class_remap_lab = DATA_LABELS["learning_map"]
  class_inv_remap_lab = DATA_LABELS["learning_map_inv"]
  class_ignore_lab = DATA_LABELS["learning_ignore"]
  nr_classes_lab = len(class_inv_remap_lab)

  class_strings_pred = DATA_PRED["labels"]
  class_remap_pred = DATA_PRED["mos_semantic_map"]
  class_inv_remap_pred = DATA_PRED["mos_semantic_map_inv"]
  nr_classes_pred = len(class_remap_pred)

  # make lookup table for mapping
  maxkey_lab = max(class_remap_lab.keys())
  maxkey_pred = max(class_remap_pred.keys())
  
  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut_lab = np.zeros((maxkey_lab + 100), dtype=np.int32)
  remap_lut_lab[list(class_remap_lab.keys())] = list(class_remap_lab.values())

  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut_pred = np.zeros((maxkey_pred + 100), dtype=np.int16)
  remap_lut_pred[list(class_remap_pred.keys())] = list(class_remap_pred.values())

  # create evaluator
  ignore = []
  for cl, ign in class_ignore_lab.items():
    if ign:
      x_cl = int(cl)
      ignore.append(x_cl)
      print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

  # create evaluator
  if FLAGS.backend == "torch":
    from auxiliary.torch_ioueval import iouEval
    evaluator = iouEval(nr_classes_lab, ignore)
  if FLAGS.backend == "numpy":
    from auxiliary.np_ioueval import iouEval
    evaluator = iouEval(nr_classes_lab, ignore)
  else:
    print("Backend for evaluator should be one of ", str(backends))
    quit()
  evaluator.reset()

  # get test set
  test_sequences = DATA_LABELS["split"][FLAGS.split]
  print('{}{}'.format('Test sequences: ', test_sequences))

  # get label paths
  label_names = []
  for sequence in test_sequences:
    #sequence = '2013_05_28_drive_%04d_sync' %sequence #KITTI-360
    sequence = '{0:02d}'.format(int(sequence)) # KITTI odometry
    label_paths = os.path.join(FLAGS.dataset, 'sequences', str(sequence), "labels")
    # populate the label names
    seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn if ".label" in f]
    seq_label_names.sort()
    label_names.extend(seq_label_names)
  # print(label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    #sequence = '2013_05_28_drive_%04d_sync' %sequence #KITTI-360
    sequence = '{0:02d}'.format(int(sequence)) #KITTI odometry
    pred_paths = os.path.join(FLAGS.predictions, str(sequence), "predictions")
    # populate the label names
    seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(pred_paths)) for f in fn if ".bin" in f]
    seq_pred_names.sort()
    pred_names.extend(seq_pred_names)
  # print(pred_names)

  # check that I have the same number of files
  print("labels: ", len(label_names))
  print("predictions: ", len(pred_names))
  assert(len(label_names) == len(pred_names))

  progress = 10
  count = 0
  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison
  for label_file, pred_file in zip(label_names[:], pred_names[:]):
    count += 1
    if 100 * count / len(label_names) > progress:
      print("{:d}% ".format(progress), end="", flush=True)
      progress += 10

    # print("evaluating label ", label_file)
    # open label
    label = np.fromfile(label_file, dtype=np.int32) # label SemanticKITTI
    label = label.reshape((-1))  # reshape to vector
    label = label & 0xFFFF       # get lower half for semantics
    label = remap_lut_lab[label]       # remap to xentropy format

    # open prediction
    pred = np.fromfile(pred_file, dtype=np.int16)
    pred = pred.reshape((-1))    # reshape to vector
    pred = remap_lut_pred[pred]       # remap to xentropy format

    assert(len(label) == len(pred))
    
    # add single scan to evaluation
    evaluator.addBatch(pred, label)

  # when I am done, print the evaluation
  m_accuracy = evaluator.getacc()
  m_jaccard, class_jaccard = evaluator.getIoU()

  print('{split} set:\n'
    'Acc avg {m_accuracy:.3f}\n'
    'IoU avg {m_jaccard:.3f}'.format(split=splits,
                                    m_accuracy=m_accuracy,
                                    m_jaccard=m_jaccard))

  save_to_log(FLAGS.predictions,'scores.txt','{}{}{}{}'.format('Evaluation on SemanticKitti motion labels:\n','sequences: ', test_sequences, '\nWith outliers as static!'))

  save_to_log(FLAGS.predictions,'scores.txt','{split} set:\n'
    'Acc avg {m_accuracy:.3f}\n'
    'IoU avg {m_jaccard:.3f}'.format(split=splits,
                                    m_accuracy=m_accuracy,
                                    m_jaccard=m_jaccard))
  
  # print also classwise
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
        i=i, class_str=class_strings_lab[class_inv_remap_lab[i]], jacc=jacc))
      save_to_log(FLAGS.predictions, 'scores.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
          i=i, class_str=class_strings_lab[class_inv_remap_lab[i]], jacc=jacc))
  
  '''
  # print for spreadsheet
  print("*" * 80)
  print("below can be copied straight for paper table")
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      if int(class_inv_remap[i]) > 1:
        sys.stdout.write('iou_moving: {jacc:.3f}'.format(jacc=jacc.item()))
  sys.stdout.write('\n')
  sys.stdout.flush()

  # if codalab is necessary, then do it
  # for moving object detection, we only care about the results of moving objects
  if FLAGS.codalab is not None:
    results = {}
    for i, jacc in enumerate(class_jaccard):
      if i not in ignore:
        if int(class_inv_remap[i]) > 1:
          results["iou_moving"] = float(jacc)
    # save to file
    output_filename = os.path.join(FLAGS.codalab, 'scores.txt')
    with open(output_filename, 'w') as yaml_file:
      yaml.dump(results, yaml_file, default_flow_style=False)
  '''
