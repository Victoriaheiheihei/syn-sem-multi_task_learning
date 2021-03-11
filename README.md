# syn-sem-multi_task_learning

This repository contains the code for our paper "基于联合学习的成分句法与AMR语义分析方法".

The code is developed under Pytorch '1.1.0' Due to the compitibility reason of Pytorch, it may not be loaded by some lower version (such as 0.4.0).

Please create issues if there are any questions! This can make things more tractable.

#About AMR
AMR is a graph-based semantic formalism, which can unified representations for several sentences of the same meaning. Comparing with other structures, such as dependency and semantic roles, the AMR graphs have several key differences:

AMRs only focus on concepts and their relations, so no function words are included. Actually the edge labels serve the role of function words.
Inflections are dropped when converting a noun, a verb or named entity into a AMR concept. Sometimes a synonym is used instead of the original word. This makes more unified AMRs so that each AMR graph can represent more sentences.
Relation tags (edge labels) are predefined and are not extracted from text (like the way OpenIE does). More details are in the official AMR page [AMR](https://amr.isi.edu/)


