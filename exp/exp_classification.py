from data_provider.data_factory import data_provider, load_dataset
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv

warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        Data = load_dataset(self.args, self.args.data_path)

        self.train_data, self.train_loader  = data_provider(self.args, Data['train_data'], Data['train_labels'], Data['train_class_names'], Data['train_max_len'], shuffle_flag=True)
        self.val_data,   self.val_loader    = data_provider(self.args, Data['val_data'], Data['val_labels'], Data['val_class_names'], Data['val_max_len'], shuffle_flag=True)
        self.test_data,  self.test_loader   = data_provider(self.args, Data['test_data'], Data['test_labels'], Data['test_class_names'], Data['test_max_len'], shuffle_flag=False)

        self.args.seq_len   = max(self.train_data.max_seq_len, self.test_data.max_seq_len)
        self.args.enc_in    = self.train_data.feature_df.shape[1]
        self.args.num_class = len(self.train_data.class_names)

        print('seq_len: ', self.args.seq_len)
        print('enc_in: ', self.args.enc_in)
        print('num_class: ', self.args.num_class)

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                padding_mask = padding_mask.to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                padding_mask = padding_mask.to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(self.val_data, self.val_loader, criterion)

            early_stopping(-val_accuracy, self.model, self.args.checkpoints)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} best Vali Acc: {5}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, early_stopping.best_score))
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(self.args.checkpoints, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                # padding_mask = padding_mask.float().to(self.device)
                padding_mask = padding_mask.to(self.device)

                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        print('test accuracy:{}'.format(accuracy))

        save_txt = self.args.order_input

        # Creates the file if it doesn't exist
        results_csv = os.path.join(self.args.results_path, self.args.problem + '.csv')
        if not os.path.exists(results_csv):
            print('creating the result file ...')
            with open(results_csv, 'w', newline='') as f:
                csv_write = csv.writer(f)
                csv_head = ["time", "train_id", "test accuracy", "setting", "save_txt"]
                csv_write.writerow(csv_head)

        # Append the results to the file
        with open(results_csv, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [time.ctime(), self.args.train_id, accuracy, setting, save_txt]
            csv_write.writerow(data_row)
            print('save the results to: ', self.args.results_path)

        # Creates the file if it doesn't exist
        results_summary_csv = os.path.join(os.path.join('./results', self.args.train_id + '_' + self.args.model), 'summary.csv')
        if not os.path.exists(results_summary_csv):
            print('creating the summary file ...')
            with open(results_summary_csv, 'w', newline='') as f:
                csv_write = csv.writer(f)
                csv_head = ["time", "train_id", "test accuracy", "setting", "save_txt"]
                csv_write.writerow(csv_head)

        # Append the results to the file
        with open(results_summary_csv, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            data_row = [time.ctime(), self.args.train_id, accuracy, setting, save_txt]
            csv_write.writerow(data_row)
            print('save the results to: ', os.path.join('./results', self.args.train_id + '_' + self.args.model))

        return
