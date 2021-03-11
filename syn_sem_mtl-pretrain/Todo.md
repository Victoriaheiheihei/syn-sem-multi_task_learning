# 联合学习模型

v1 版本为使用task embedding区分任务，v2 版本使用task token区分
---

Task1 & Task2 
暂定Task1 为 NMT 任务, Task2 为 gen-AMR 任务


opts.joint_preprocess_opts(parser)



opt warmup step 2 task参数设置，很多

loss也需要

trainer也需要

问题来了，如何3个task该如何拓展呢？可以自动拓展吗？
hasattr?


要改这里report_stats

1. share_vocab是共享 en amr 的词表，即 src, tgt2的词表，与 tgt无关。share_embedding同理

2. save_checkpoint 2次是不是不太好


    while task_step <= train_steps or task2_step <= train_steps2:
      self.save = False
      # self.save = True
      if task_step <= train_steps:
        task_step = self.train_task(task_step, train_steps, valid_steps,task_type='task')
      # self.save = False
      # self.save = True
      if task2_step <= train_steps2:
        task2_step = self.train_task(task2_step, train_steps2, valid_steps,task_type='task2')

trainer测试一下，先保留强，还是后保留比较好