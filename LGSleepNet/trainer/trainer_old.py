import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn

selected_d = {"outs": [], "trg": [],"probs": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.device = torch.device('cuda:0')
        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
        self.adjustment = self.compute_adjustment(self.data_loader,tro=2)

    def compute_adjustment(self, data_loader, tro):
        """compute the base probabilities"""

        label_freq = {}
        for i, (inputs, target) in enumerate(data_loader):
            target = target.to(self.device)
            for j in target:
                key = int(j.item())
                label_freq[key] = label_freq.get(key, 0) + 1
        #print("q",label_freq)
        label_freq = dict(sorted(label_freq.items()))
        #print("q1", label_freq)
        label_freq_array = np.array(list(label_freq.values()))
        label_freq_array = label_freq_array / label_freq_array.sum()
        adjustments = np.log(label_freq_array ** tro + 1e-12)
        #print("q3", adjustments)
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.to(self.device)
        return adjustments

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        overall_probs = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data).to(self.device)
            #output = output.to(self.device)
            #output = output + self.adjustment
            #loss = self.criterion(output, target, self.class_weights, self.device)
            #print(output.shape)
            #print(target.shape)
            #self.criterion.set_epsilon(A)
            loss = self.criterion(output, target).to(self.device)


            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                #print(self.criterion.epsilon.grad)
                # 查看 epsilon 的梯度
                #epsilon_grad = self.criterion.epsilon.grad.item()
                #print(epsilon_grad)

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs, probs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
                selected_d["probs"] = probs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])
                overall_probs.extend(selected_d["probs"])
            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001


        return log, overall_outs, overall_trgs, overall_probs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            probs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output= self.model(data).to(self.device)
                #output = output.to(self.device)
                #output = output + self.adjustment
                #loss = self.criterion(output, target, self.class_weights, self.device)
                #self.criterion.set_epsilon(A)
                loss = self.criterion(output, target).to(self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                probs = np.append(probs, output.data.cpu().numpy())
                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs,probs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
