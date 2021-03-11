# syn-sem-multi_task_learning

本仓库中包含了文章 "基于联合学习的成分句法与AMR语义分析方法"中涉及的部分代码

该代码是在Pytorch'1.1.0'下开发的。由于Pytorch的兼容性原因，某些较低版本（例如0.4.0）可能不能成功运行代码。

如有任何疑问，请提出问题！

## 关于 AMR
AMR是基于图的语义表示，可以统一表示具有相同含义的多个句子。 与依赖关系和语义角色等其他结构相比，AMR图具有几个主要差异：AMR仅关注概念及其关系，因此不包括功能词；边缘标签充当功能词的角色；将名词，动词或命名实体转换为AMR概念时，将不使用拐点； 有时使用同义词代替原始单词。 这样可以使AMR更加统一，以便每个AMR图可以表示更多的句子。更多细节内容可以访问AMR官方网址 [AMR](https://amr.isi.edu/)

## 关于成分句法分析
成分句法树是一种基于树的句法表示，它包含输入句子中每个单词的词性以及句法成分。更多详情请见 [Constituency](https://catalog.ldc.upenn.edu/LDC99T42)

## 数据

因为AMR解析语料和成分句法语料无法公开，因此此处并未上传，但用于预训练的机器翻译WMT14英德语料可以在链接中进行下载[WMT14](http://www.statmt.org/wmt14/translation-task.html)。在实验结果分析中，我们利用成分句法分析任务上性能最好的模型生成自动标注句法树，以比较用于预训练的自动标注句法树准确性对后续实验的影响，该模型可以在原作者提供的链接中进行下载[LAL-Parser](https://github.com/KhalilMrini/LAL-Parser)

### 预处理
1.线性化
  使用先序遍历将树或图结构转化成线性化结构。
3.BPE/BERT tokenize
  使用[Subword-nmt](https://github.com/rsennrich/subword-nmt.git)工具对文本进行子词化处理，或使用[BertTokenizer](https://github.com/huggingface/transformers)对文本进行切分


## 模型

本文使用 Open-NMT 作为 Transformer 模型的实现。我们上传了两个版本的模型，一个是基于单任务的Transformer模型——single_task，另一个是基于多任务的Transformer模型——multi_task。相较于单任务的模型，多任务模型的底层结构与单任务一致，即保持传统Transformer的Encoder和Decoder层，但是在参数设置上，我们增加了task_type(['task','task2'])用于区分在训练过程中不同任务的参数更新。两个模型最本质的区别在于训练策略上的不同，多任务模型每训练一步就更换成另一任务训练集进行训练，为此我们在原参数基础上增加了train_steps2、warmup_steps2、learning_rate2、batch_size2作为第二个任务的参数。

