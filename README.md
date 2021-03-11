# syn-sem-multi_task_learning

本仓库中包含了文章 "基于联合学习的成分句法与AMR语义分析方法"中涉及的部分代码

该代码是在Pytorch'1.1.0'下开发的。由于Pytorch的兼容性原因，某些较低版本（例如0.4.0）可能不会加载该代码。

如有任何疑问，请提出问题！

## 关于 AMR
AMR是基于图的语义表示，可以统一表示具有相同含义的多个句子。 与依赖关系和语义角色等其他结构相比，AMR图具有几个主要差异：AMR仅关注概念及其关系，因此不包括功能词；边缘标签充当功能词的角色；将名词，动词或命名实体转换为AMR概念时，将不使用拐点； 有时使用同义词代替原始单词。 这样可以使AMR更加统一，以便每个AMR图可以表示更多的句子。更多细节内容可以访问AMR官方网址 [AMR](https://amr.isi.edu/)

## 关于成分句法分析
成分句法树是一种基于树的句法表示，它包含输入句子中每个单词的词性以及句法成分。更多详情请见 [Constituency](https://catalog.ldc.upenn.edu/LDC99T42)

## Data

因为AMR解析语料和成分句法语料无法公开，因此此处并未上传，但用于预训练的机器翻译WMT14英德语料可以在链接中进行下载[WMT14](http://www.statmt.org/wmt14/translation-task.html)。在实验结果分析中，我们利用成分句法分析任务上性能最好的模型生成自动标注句法树，以比较用于预训练的自动标注句法树准确性对后续实验的影响，该模型可以在原作者提供的链接中进行下载[LAL-Parser](https://github.com/KhalilMrini/LAL-Parser)

## Model

本文使用 Open-NMT 作为 Transformer 模型的实现于We provide two models, one is a model based on single task, and the other is a model based on joint learning.The main improvement lies in

