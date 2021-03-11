import argparse
import codecs
from utils.parse_tree import penn_treebank_reader

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-parse_file', required=True)
  parser.add_argument('-graph_file', required=True)

  opt = parser.parse_args()

  return opt

def extract_token_edge(tree):
  tokens = []
  edegs = []
  
  def tranverse(node, node_index):
    if node is None:
      return
    
    tokens.append(node.text)
    
    for child_index, child in enumerate(node.children):
      tranverse(child, node_index + child_index + 1)
      edegs.append((node_index, child_index + 1, 'd'))
      edegs.append((child_index + 1, node_index, 'r'))
      
  tranverse(tree.root)
  for i in range(len(tokens)):
    edegs.append((i, i, 's'))
    

def main():
  opt = parse_args()
  print("[Info] {}".format(opt))
  print("[Info] Reading trees...")
  tree_reader = penn_treebank_reader(opt.parse_file, False, True, False, False)
  out_file1 = codecs.open(opt.graph_file + ".tok", 'w+', 'utf-8')
  out_file2 = codecs.open(opt.graph_file + ".graph", 'w+', 'utf-8')
  for tree in tree_reader:
    tokens, edegs = extract_token_edge(tree)
    out_file1.write(" ".join(tokens))
    out_file1.write("\n")
    for edge in edegs:
      out_file2.write("(%d,%d,%s) " % (edge[0], edge[1], edge[2]))
  out_file1.close()
  out_file2.close()

if __name__ == "__main__":
  main()