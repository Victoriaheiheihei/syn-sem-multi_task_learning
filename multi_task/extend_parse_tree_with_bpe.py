"""
given a word sequence and its parse tree, and its BPEed sequence, return the BPEed parse tree

if a word is segmented into multiple subwords, then every subword maps to a terminal node, and 
                                                    they share same pos node
"""
import argparse
import codecs
from utils.parse_tree import penn_treebank_reader, TreeNode

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-parse_file', required=True)
  parser.add_argument('-bpe_file', required=True)
  parser.add_argument('-output_file', required=True)

  opt = parser.parse_args()

  return opt

def extend_bpe(tree, subwords):
  """
  """
  words = [n.text for n in tree.terminal]
  maps = []
  i, j = 0, 0
  merged_word = ""
  curr_start = 0
  while i < len(words) and j < len(subwords):
    if subwords[j].endswith("@@"):
      sub = subwords[j][0:-2]
    else:
      sub = subwords[j]
    merged_word += sub
    if merged_word.lower() == words[i].lower():
      curr_end = j
      maps.append((curr_start, curr_end))
      i += 1
      curr_start = j + 1
      merged_word = ""
    j += 1
    
  assert i == len(words) and j == len(subwords)
  
  for node, (start, end) in zip(tree.terminal, maps):
    pre = node.parent
    pre.remove_child_at(0)
    del node
    for i in range(start, end + 1):
      subword_node = TreeNode(subwords[i])
      pre.add_child(subword_node)
      
  return tree

def main():
  opt = parse_args()
  print("[Info] {}".format(opt))
  print("[Info] Reading trees...")
  
  out_file = codecs.open(opt.output_file, 'w+', 'utf-8')
  count = 0
  with open(opt.bpe_file, "r") as f:
    for tree, line in zip(penn_treebank_reader(opt.parse_file), f):
      tree = extend_bpe(tree, line.strip().split())
      count += 1
      if count % 5000 == 0:
        print("[Info] Reading {} trees...".format(count))
      out_file.write(tree.root.to_string() + "\n")
  out_file.close()

if __name__ == "__main__":
  main()
  