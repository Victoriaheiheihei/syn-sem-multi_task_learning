from sys import argv
import torch
import glob
import re
import numpy as np
import gc


def get_checkpoints(path):
  #get checkpints from model save dir
  path_list = glob.glob(path + '/' + '*.pt')
  return path_list

def sort_checkpoints(path_list):
  iter_list = []
  saved_models = []
  for path in path_list:
    # path: model_step_100000.pt
    p = re.compile(r'model_step_(.*)\.pt$')
    matches = p.findall(path)
    if len(matches) > 0:
      i = int(matches[0])
      saved_models.append(path)
      iter_list.append(i)
  sorted_index = np.argsort(iter_list)
  path_list = [saved_models[i] for i in sorted_index]
  return path_list

if __name__=="__main__":
  script, model_save_dir, ensemble_path , number = argv
  if model_save_dir is None:
    print("model_save_dir error")

  checkpoints_list = get_checkpoints(model_save_dir)
  checkpoints_list = sort_checkpoints(checkpoints_list)
  checkpoints_list = checkpoints_list[-int(number):]
  print("Averaging checkpoints: \n{}".format(checkpoints_list))
 
  print("start average the last {} model".format(number)) 

  model_dict = {}
  generator_dict = {}

  for checkpoint_path in checkpoints_list:
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    for key, value in checkpoint['model'].items():
      value_sum = model_dict.get(key, 0)
      value_sum += value
      model_dict[key] = value_sum
    for key, value in checkpoint['generator'].items():
      value_sum = generator_dict.get(key, 0)
      value_sum += value
      generator_dict[key] = value_sum

    if checkpoint_path == checkpoints_list[-1]:
      vocab = checkpoint['vocab']
      opt = checkpoint['opt']
      optim = checkpoint['optim']
    else:
      del checkpoint
      gc.collect()
      torch.cuda.empty_cache()
  
  model_dict_avg = {}
  generator_dict_avg = {}
  for key, value in model_dict.items():
    model_dict_avg[key] = value / int(number)
  for key, value in generator_dict.items():
    generator_dict_avg[key] = value / int(number)

  
  checkpoint_ensemble = {'vocab':vocab, 'opt': opt, 'model': model_dict_avg, 'generator':generator_dict_avg, 'optim':optim}
  torch.save(checkpoint_ensemble, ensemble_path)
  print("ensemble end and save model to {}".format(ensemble_path))