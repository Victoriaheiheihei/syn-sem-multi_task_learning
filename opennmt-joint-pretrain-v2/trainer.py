"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from utils.loss import build_loss_compute
from utils.logging import logger
from utils.report_manager import build_report_manager
from utils.statistics import Statistics
from utils.distributed import all_gather_list, all_reduce_and_rescale_tensors
from inputters.dataset import make_features
import torch


def build_joint_trainer(opt, device_id, model, fields, optim, 
                        train_iter_fct, valid_iter_fct,
                        train_iter_fct2, valid_iter_fct2,
                        model_saver=None):
  """
  Simplify `Trainer` creation based on user `opt`s*

  Args:
      opt (:obj:`Namespace`): user options (usually from argument parsing)
      model (:obj:`onmt.models.NMTModel`): the model to train
      fields (dict): dict of fields
      optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
      model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
          used to save the model
  """
  train_loss = build_loss_compute(
    model, fields["tgt"].vocab, opt, task_type='task')
  valid_loss = build_loss_compute(
    model, fields["tgt"].vocab, opt, train=False, task_type='task')

  train_loss2 = build_loss_compute(
    model, fields["tgt2"].vocab, opt, task_type='task2')
  valid_loss2 = build_loss_compute(
    model, fields["tgt2"].vocab, opt, train=False, task_type='task2')
  

  trunc_size = opt.truncated_decoder  # Badly named...
  shard_size = opt.max_generator_batches
  norm_method = opt.normalization
  grad_accum_count = opt.accum_count
  n_gpu = opt.world_size
  if device_id >= 0:
    gpu_rank = opt.gpu_ranks[device_id]
  else:
    gpu_rank = 0
    n_gpu = 0
  gpu_verbose_level = opt.gpu_verbose_level

  report_manager = build_report_manager(opt, 'task')
  report_manager2 = build_report_manager(opt, 'task2')


  trainer = JointedTrainer(model, 
                      train_iter_fct, valid_iter_fct, train_loss, valid_loss,
                      train_iter_fct2, valid_iter_fct2, train_loss2, valid_loss2,
                      optim, trunc_size,
                      shard_size, norm_method,
                      grad_accum_count, n_gpu, gpu_rank,
                      gpu_verbose_level, 
                      report_manager, report_manager2, 
                      model_saver=model_saver)
  return trainer


class JointedTrainer(object):
  """
  Jointly Learning Trainner
  """

  def __init__(self, model,
                train_iter_fct, valid_iter_fct, train_loss, valid_loss,
                train_iter_fct2, valid_iter_fct2, train_loss2, valid_loss2,
                optim, trunc_size=0, 
                shard_size=32, norm_method="sents", 
                grad_accum_count=1, n_gpu=1, gpu_rank=1,
                gpu_verbose_level=0, 
                report_manager=None, report_manager2=None,
                model_saver=None):
    # Basic attributes.
    self.model = model
    self.train_iter_fct = train_iter_fct
    self.valid_iter_fct = valid_iter_fct
    self.train_loss = train_loss
    self.valid_loss = valid_loss

    self.train_iter_fct2 = train_iter_fct2
    self.valid_iter_fct2 = valid_iter_fct2
    self.train_loss2 = train_loss2
    self.valid_loss2 = valid_loss2

    self.optim = optim
    self.trunc_size = trunc_size
    self.shard_size = shard_size
    self.norm_method = norm_method
    self.grad_accum_count = grad_accum_count
    self.n_gpu = n_gpu
    self.gpu_rank = gpu_rank
    self.gpu_verbose_level = gpu_verbose_level
    self.report_manager = report_manager
    self.report_manager2 = report_manager2
    self.model_saver = model_saver

    assert grad_accum_count > 0
    if grad_accum_count > 1:
      assert(self.trunc_size == 0), \
        """To enable accumulated gradients,
           you must disable target sequence truncating."""

    # Set model in training mode.
    self.model.train()

  def train(self, train_steps, train_steps2, valid_steps):

    logger.info('Start training...')

    task_step = self.optim._task_step + 1
    task2_step = self.optim._task2_step + 1

    self.train_iter = self.get_task_batch(task_step, task_type='task')
    self.train_iter2 = self.get_task_batch(task2_step, task_type='task2')
    
    self.total_stats = Statistics(task_type='task')
    self.report_stats = Statistics(task_type='task')
    self._start_report_manager(self.report_manager, start_time=self.total_stats.start_time)

    self.total_stats2 = Statistics(task_type='task2')
    self.report_stats2 = Statistics(task_type='task2')
    self._start_report_manager(self.report_manager2, start_time=self.total_stats2.start_time)

    while task_step <= train_steps or task2_step <= train_steps2:
      self.save = False
      # self.save = True
      if task_step <= train_steps:
        task_step = self.train_task(task_step, train_steps, valid_steps,task_type='task')
      # self.save = False
      # self.save = True
      if task2_step <= train_steps2:
        task2_step = self.train_task(task2_step, train_steps2, valid_steps,task_type='task2')
        

    return self.total_stats, self.total_stats2

  def get_task_batch(self, step, task_type='task'):
    if task_type == 'task':
      while True:
        train_iter = self.train_iter_fct()
        for i, batch in enumerate(train_iter):
          step += 1
          yield i, batch
        if self.gpu_verbose_level > 0:
          logger.info('GpuRank %d: we completed an epoch \
                      at step %d' % (self.gpu_rank, step))
    else:
      while True:
        train_iter = self.train_iter_fct2()
        for i, batch in enumerate(train_iter):
          step += 1
          yield i, batch
        if self.gpu_verbose_level > 0:
          logger.info('GpuRank %d: we completed an epoch \
                      at step %d' % (self.gpu_rank, step))

  def train_task(self, step, train_steps, valid_steps, task_type):
    true_batchs = []
    accum = 0
    normalization = 0
    reduce_counter = 0

    while True:
        if task_type == 'task':
          i, batch = next(self.train_iter)
        else:          
          i, batch = next(self.train_iter2)

        if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
          if self.gpu_verbose_level > 1:
            logger.info("GpuRank %d: index: %d accum: %d"
                        % (self.gpu_rank, i, accum))

          true_batchs.append(batch)

          if self.norm_method == "tokens":
            if task_type == 'task':
              num_tokens = batch.tgt[1:].ne(self.train_loss.padding_idx).sum()
            else:
              num_tokens = batch.tgt2[1:].ne(self.train_loss2.padding_idx).sum()
            normalization += num_tokens.item()
          else:
            if task_type == 'task':
              normalization += batch.batch_size
            else:
              normalization += batch.batch_size2
          accum += 1
          if accum == self.grad_accum_count:
            reduce_counter += 1
            if self.gpu_verbose_level > 0:
              logger.info("GpuRank %d: reduce_counter: %d \
                          n_minibatch %d"
                          % (self.gpu_rank, reduce_counter,
                             len(true_batchs)))
            if self.n_gpu > 1:
              normalization = sum(all_gather_list
                                    (normalization))

            if task_type == 'task':
              self._gradient_accumulation(
                true_batchs, normalization, self.total_stats,
                self.report_stats, task_type=task_type)

              self.report_stats = self._maybe_report_training(
                self.report_manager,
                step, train_steps,
                self.optim.learning_rate,
                self.report_stats)
            else:
              self._gradient_accumulation(
                true_batchs, normalization, self.total_stats2,
                self.report_stats2, task_type=task_type)

              self.report_stats2 = self._maybe_report_training(
                self.report_manager2,
                step, train_steps,
                self.optim.learning_rate,
                self.report_stats2)

            true_batchs = []
            accum = 0
            normalization = 0
            if (step % valid_steps == 0):
              if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: validate step %d'
                              % (self.gpu_rank, step))

              if task_type == 'task':
                valid_iter = self.valid_iter_fct()
                valid_stats = self.validate(valid_iter, task_type='task')
              else:
                valid_iter = self.valid_iter_fct2()
                valid_stats = self.validate(valid_iter, task_type='task2')

              if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: gather valid stat \
                              step %d' % (self.gpu_rank, step))
              valid_stats = self._maybe_gather_stats(valid_stats)
              if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: report stat step %d'
                              % (self.gpu_rank, step))
              
              if task_type == 'task':
                self._report_step(
                                self.report_manager,
                                self.optim.learning_rate,
                                step, valid_stats=valid_stats)
              else:
                self._report_step(
                                self.report_manager2,
                                self.optim.learning_rate,
                                step, valid_stats=valid_stats)
                
            if self.gpu_rank == 0:
              if self.save == False:
                self._maybe_save(step)
                self.save = True
            step += 1
            return step

  def validate(self, valid_iter, task_type='task'):
    """ Validate model.
        valid_iter: validate data iterator
    Returns:
        :obj:`nmt.Statistics`: validation loss statistics
    """
    # Set model in validating mode.
    self.model.eval()

    stats = Statistics(task_type=task_type)
    with torch.no_grad():
      for batch in valid_iter:
        src = make_features(batch, 'src')
        _, src_lengths = batch.src

        if task_type == 'task':
          tgt = make_features(batch, 'tgt')
        else:
          tgt = make_features(batch, 'tgt2')

        # F-prop through the model.
        outputs, attns = self.model(src, tgt, src_lengths, task_type=task_type)

        # Compute loss.
        if task_type == 'task':
          batch_stats = self.valid_loss.monolithic_compute_loss(
            batch, outputs, attns)
        else:
          batch_stats = self.valid_loss2.monolithic_compute_loss(
            batch, outputs, attns)

        # Update statistics.
        stats.update(batch_stats)

      # Set model back to training mode.
    self.model.train()

    return stats

  def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                             report_stats, task_type='task'):
      if self.grad_accum_count > 1:
          self.model.zero_grad()

      for batch in true_batchs:
          if task_type == 'task':
            target_size = batch.tgt.size(0)
          else:
            target_size = batch.tgt2.size(0)
          # Truncated BPTT: reminder not compatible with accum > 1
          if self.trunc_size:
              trunc_size = self.trunc_size
          else:
              trunc_size = target_size

          # dec_state = None
          src = make_features(batch, 'src')
          _, src_lengths = batch.src

          if task_type == 'task':
            tgt_outer = make_features(batch, 'tgt')
          else:
            tgt_outer = make_features(batch, 'tgt2')

          for j in range(0, target_size-1, trunc_size):
              # 1. Create truncated target.
              tgt = tgt_outer[j: j + trunc_size]

              # 2. F-prop all but generator.
              if self.grad_accum_count == 1:
                  self.model.zero_grad()
              outputs, attns = \
                  self.model(src, tgt, src_lengths, task_type=task_type)

              # 3. Compute loss in shards for memory efficiency.
              if task_type == 'task':
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)
              else:
                batch_stats = self.train_loss2.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)
              
              total_stats.update(batch_stats)
              report_stats.update(batch_stats)



              # 4. Update the parameters and statistics.
              if self.grad_accum_count == 1:
                  # Multi GPU gradient gather
                  if self.n_gpu > 1:
                      grads = [p.grad.data for p in self.model.parameters()
                               if p.requires_grad
                               and p.grad is not None]
                      all_reduce_and_rescale_tensors(
                          grads, float(1))
                  if task_type == 'task':
                    self.optim.step(task_type='task')
                  else:
                    self.optim.step(task_type='task2')

              # If truncated, don't backprop fully.
              # TO CHECK
              # if dec_state is not None:
              #    dec_state.detach()
              if self.model.decoder.state is not None:
                  self.model.decoder.detach_state()

      # in case of multi step gradient accumulation,
      # update only after accum batches
      if self.grad_accum_count > 1:
          if self.n_gpu > 1:
              grads = [p.grad.data for p in self.model.parameters()
                       if p.requires_grad
                       and p.grad is not None]
              all_reduce_and_rescale_tensors(
                  grads, float(1))
          if task_type == 'task':
            self.optim.step(task_type='task')
          else:
            self.optim.step(task_type='task2')

  def _start_report_manager(self, report_manager, start_time=None):
      """
      Simple function to start report manager (if any)
      """
      if report_manager is not None:
          if start_time is None:
              report_manager.start()
          else:
              report_manager.start_time = start_time

  def _maybe_gather_stats(self, stat):
      """
      Gather statistics in multi-processes cases

      Args:
          stat(:obj:onmt.utils.Statistics): a Statistics object to gather
              or None (it returns None in this case)

      Returns:
          stat: the updated (or unchanged) stat object
      """
      if stat is not None and self.n_gpu > 1:
          return Statistics.all_gather_stats(stat)
      return stat

  def _maybe_report_training(self, report_manager, step, num_steps, learning_rate,
                             report_stats):
      """
      Simple function to report training stats (if report_manager is set)
      see `onmt.utils.ReportManagerBase.report_training` for doc
      """
      if report_manager is not None:
          return report_manager.report_training(
              step, num_steps, learning_rate, report_stats,
              multigpu=self.n_gpu > 1)

  def _report_step(self, report_manager, learning_rate, step, train_stats=None,
                   valid_stats=None):
      """
      Simple function to report stats (if report_manager is set)
      see `onmt.utils.ReportManagerBase.report_step` for doc
      """
      if report_manager is not None:
          return report_manager.report_step(
              learning_rate, step, train_stats=train_stats,
              valid_stats=valid_stats)

  def _maybe_save(self, step):
      """
      Save the model if a model saver is set
      """
      if self.model_saver is not None:
          self.model_saver.maybe_save(step)