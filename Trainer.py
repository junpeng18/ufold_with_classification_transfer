import torch
import math
import os
import time
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.data_utils import get_cut_len
from Network import FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_contact_data(data, target, seq_lens=None):
    # data: (B, C, L, L)
    # target: (B, L, L)
    if seq_lens is not None:
        max_seq_len = get_cut_len(max(64, seq_lens.max().item()))
        data = data[..., :max_seq_len, :max_seq_len]
        target = target[:, :max_seq_len, :max_seq_len]
    return data.to(device), target.to(device)


def prepare_classify_data(data, target, seq_lens=None):
    # data: (B, C, L, L)
    # target: (B, )
    if seq_lens is not None:
        max_seq_len = get_cut_len(max(64, seq_lens.max().item()))
        data = data[..., :max_seq_len, :max_seq_len]
    return data.to(device), target.to(device)


class BasicTrainer(metaclass=ABCMeta):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, args, logger, lr_scheduler=None):
        super(BasicTrainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.val_per_epoch = len(val_loader)
        self.writer = SummaryWriter(self.args['tb_dir'])
        if not os.path.exists(self.args['log_dir']):
            os.makedirs(self.args['log_dir'])
        self.best_path = os.path.join(self.args['log_dir'], 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args['log_dir'], 'loss.png')

    @abstractmethod
    def val_epoch(self, epoch):
        pass

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    def train(self):
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in tqdm(range(1, self.args['epochs'] + 1)):
            train_epoch_loss, val_epoch_loss = self.train_epoch(epoch)
            self.writer.add_scalar("loss/train", train_epoch_loss, epoch)
            self.writer.add_scalar("loss/val", val_epoch_loss, epoch)
            print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e5:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if self.val_loader is None:
                val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                self.logger.info(f"not_improved_count: {not_improved_count}")
                best_state = False
            # early stop
            if not_improved_count == self.args['early_stop']:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.args['early_stop']))
                break
            # save the best state
            if best_state:
                self.save_checkpoint()
            # plot loss figure
            if self.args['plot'] == True:
                self._plot_line_figure([train_loss_list, val_loss_list], path=self.loss_figure_path)
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format(training_time / 60))
        self.writer.close()

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def _plot_line_figure(losses, path):
        train_loss = losses[0]
        val_loss = losses[1]
        plt.style.use('ggplot')
        epochs = list(range(1, len(train_loss) + 1))
        plt.plot(epochs, train_loss, 'r-o')
        plt.plot(epochs, val_loss, 'b-o')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(path, bbox_inches="tight")


class UFoldTrainer(BasicTrainer):
    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target, seq_lens, _) in enumerate(self.val_loader):
                data, target = prepare_contact_data(data, target, seq_lens)
                output = self.model(data)
                loss = self.loss(output, target, seq_lens)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target, seq_lens, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            data, target = prepare_contact_data(data, target, seq_lens)
            output = self.model(data)
            loss = self.loss(output, target, seq_lens)

            loss.backward()
            # add max grad clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()
            total_loss += loss.item()
            # log information
            if batch_idx % self.args['log_step'] == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.3f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.3f}'.format(epoch, train_epoch_loss))
        # learning rate decay
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # validation
        if self.val_loader == None:
            val_epoch_loss = 0
        else:
            val_epoch_loss = self.val_epoch(epoch)
        return train_epoch_loss, val_epoch_loss


class ClassifierTrainer(BasicTrainer):
    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target, seq_lens) in enumerate(self.val_loader):
                data, target = prepare_classify_data(data, target, seq_lens)
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target, seq_lens) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            data, target = prepare_classify_data(data, target, seq_lens)
            output = self.model(data)  # (B, C)
            loss = self.loss(output, target)

            loss.backward()
            # add max grad clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()
            total_loss += loss.item()
            # log information
            if batch_idx % self.args['log_step'] == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.3f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.3f}'.format(epoch, train_epoch_loss))
        # learning rate decay
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # validation
        if self.val_loader == None:
            val_epoch_loss = 0
        else:
            val_epoch_loss = self.val_epoch(epoch)
        return train_epoch_loss, val_epoch_loss


class UFoldTransferTrainer(BasicTrainer):
    def __init__(self, model, feat_extract_model, loss, optimizer, train_loader, val_loader, args, logger, lr_scheduler=None):
        super(UFoldTransferTrainer, self).__init__(model, loss, optimizer, train_loader,
                                                   val_loader, args, logger, lr_scheduler)
        # FeatureExtractor(class_model, layers=["layer2", "layer3", "layer4"])
        self.feat_extract_model = feat_extract_model

    def val_epoch(self, epoch):
        self.model.eval()
        self.feat_extract_model.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target, seq_lens, _) in enumerate(self.val_loader):
                data, target = prepare_contact_data(data, target, seq_lens)
                feats = self.feat_extract_model(data)
                feat_l2, feat_l3, feat_l4 = feats["layer2"], feats["layer3"], feats["layer4"]
                output = self.model(data, feat_l2, feat_l3, feat_l4)
                loss = self.loss(output, target, seq_lens)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        self.feat_extract_model.model.eval()
        total_loss = 0
        for batch_idx, (data, target, seq_lens, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            data, target = prepare_contact_data(data, target, seq_lens)
            with torch.no_grad():
                feats = self.feat_extract_model(data)
            feat_l2, feat_l3, feat_l4 = feats["layer2"], feats["layer3"], feats["layer4"]
            output = self.model(data, feat_l2, feat_l3, feat_l4)  # (B, C)
            loss = self.loss(output, target, seq_lens)

            loss.backward()
            # add max grad clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
            self.optimizer.step()
            total_loss += loss.item()
            # log information
            if batch_idx % self.args['log_step'] == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.3f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.3f}'.format(epoch, train_epoch_loss))
        # learning rate decay
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # validation
        if self.val_loader == None:
            val_epoch_loss = 0
        else:
            val_epoch_loss = self.val_epoch(epoch)
        return train_epoch_loss, val_epoch_loss
