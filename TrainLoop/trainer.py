import os
from logging import Logger
from typing import Union, Callable, List

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


logger = Logger(__name__, level='DEBUG')

class Trainer:
    ''' Custom trainer for segmentation '''
    def __init__(self,
                model: torch.nn.Module,
                save_dir: str,
                loss_fn: Callable,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                n_epochs: Union[int, None] = None,
                n_steps: Union[int, None] = None,
                n_steps_eval: Union[int, None] = None,
                device: str = 'cuda',
                early_stop_rounds: Union[int, None] = None,
                eval_strat: str = 'epoch',
                loss_lesser_is_better: bool = True,
                save_only_best: bool = False,
                accumulate_metric_callbacks: List[Callable] = [],
                compute_metric_callbacks: List[Callable] = [],
                ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if eval_strat == 'epoch':
            if n_epochs is None:
                raise AttributeError('Parameter "eval_strat" is set to "epoch" but param "n_epoch" was not provided')
            self.n_iters = n_epochs
            self.iter_name = 'Epoch'
        elif eval_strat == 'step':
            if n_steps is None:
                raise AttributeError('Parameter "eval_strat" is set to "steps" but param "n_steps" was not provided')
            self.n_iters = n_steps
            self.n_steps_eval = n_steps_eval
            self.iter_name = 'Step'
        else:
            raise AttributeError('Parameter "eval_strat" should be either "epoch" or "steps"')
            
        self.save_dir = save_dir
        self.device = device
        self.early_stop_rounds = early_stop_rounds
        self.eval_strat = eval_strat
        self.loss_lesser_is_better = loss_lesser_is_better
        self.save_only_best = save_only_best
        self.accumulate_metric_callbacks = accumulate_metric_callbacks
        self.compute_metric_callbacks = compute_metric_callbacks
        
        self.model = self.model.to(self.device)
        
        self.curr_iter = 0
        
    def _epoch_train_loop(self, dl_train: torch.utils.data.DataLoader, dl_val: torch.utils.data.DataLoader):
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_hist = []
        self.val_hist = []
        best_val_loss = {'time': f'{self.iter_name}_0_batch_0', 'value': np.inf}
        lr_scheduled = []

        self.early_stop_cnt = 0
        
        with tqdm(total=self.n_iters, desc=f'{self.iter_name}: 0', leave=True) as pbar:
            for iteration in range(self.n_iters):
                self.curr_iter = iteration
                train_hist_iter = []
                with tqdm(total=len(dl_train), desc=f'Batch: 0', leave=False) as inner_pbar:
                    for idx_batch, (X_train, y_train) in enumerate(dl_train):
                        self.optimizer.zero_grad()
                        X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                        self.model.train()
                        pred = self.model(X_train)
                        loss = self.loss_fn(pred, y_train)
                        loss.backward()

                        self.optimizer.step()
                        train_hist_iter.append(loss.item())
                        
                        inner_pbar.update(1)
                        inner_pbar.set_description(f'Batch: {idx_batch}')
                        
                        
                    best_val_loss = self.process_end_of_iter(train_hist_iter,
                                                                             dl_val,
                                                                             best_val_loss,
                                                                             lr_scheduled,
                                                                             iteration,
                                                                             idx_batch)
                    if self.early_stop_cnt >= self.early_stop_rounds:
                        # Add early_stopping callback
                        # self.es_callback()
                        print(f'Finished. Done {iteration} {self.iter_name}s. Best model learned at {best_val_loss["time"]}')
                        return self.train_hist, self.val_hist

                pbar.update(1)
                pbar.set_description(f'{self.iter_name}: {iteration}')
    
    
    def _steps_train_loop(self,  dl_train: torch.utils.data.DataLoader, dl_val: torch.utils.data.DataLoader):
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_hist = []
        self.val_hist = []
        best_val_loss = {'time': f'{self.iter_name}_0_batch_0', 'value': np.inf}
        lr_scheduled = []

        self.early_stop_cnt = 0
        
        iteration = 0
        train_hist_iter = []
        
        with tqdm(total=self.n_iters, desc=f'{self.iter_name}: {iteration}', leave=False) as pbar:
            while iteration < self.n_iters:
                self.curr_iter = iteration
                for idx_batch, (X_train, y_train) in enumerate(dl_train):
                    self.optimizer.zero_grad()
                    X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                    self.model.train()
                    pred = self.model(X_train)
                    loss = self.loss_fn(pred, y_train)
                    loss.backward()
                    
                    self.optimizer.step()
                    train_hist_iter.append(loss.item())
                    
                    if (iteration % self.n_steps_eval == 0) and (iteration != 0):
                        best_val_loss = self.process_end_of_iter(train_hist_iter,
                                                                     dl_val,
                                                                     best_val_loss,
                                                                     lr_scheduled,
                                                                     iteration,
                                                                     idx_batch)
                        train_hist_iter = []
                        if self.early_stop_cnt >= self.early_stop_rounds:
                            # Add early_stopping callback
                            # self.es_callback()
                            logger.info(f'Finished. Done {iteration} {self.iter_name}s. Best model learned at {best_val_loss["time"]}')
                            return self.train_hist, self.val_hist

                    iteration += 1
                    pbar.update(1)
                    pbar.set_description(f'{self.iter_name}: {iteration}')

        
    def train(self, dl_train: torch.utils.data.DataLoader, dl_val: torch.utils.data.DataLoader):
        if self.eval_strat == 'epoch':
            self._epoch_train_loop(dl_train, dl_val)
        else:
            self._steps_train_loop(dl_train, dl_val)
            

    def process_end_of_iter(self, train_hist_iter, dl_val, best_val_loss, lr_scheduled, iteration, idx_batch):
        self.train_hist.append(sum(train_hist_iter) / len(train_hist_iter))

        val_loss = self.validate(dl_val)
        self.val_hist.append(val_loss)

        for cb in self.compute_metric_callbacks:
            cb(save_dir=self.save_dir,
                curr_iter=self.curr_iter,
                train_loss=self.train_hist[-1],
                val_loss=self.val_hist[-1])  

        best_val_loss = self.check_for_best_model(best_val_loss, val_loss, iteration, idx_batch)

        self.scheduler.step()
        lr_scheduled.append(self.scheduler.get_last_lr())
        
        return best_val_loss

    
    def validate(self, dl_val):
        self.model.eval()
        val_hist_batch = []
        pred_hist_batch = []
        with torch.no_grad():
            for X_val, y_val in dl_val:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                pred = self.model(X_val)
                
                loss = self.loss_fn(pred, y_val)
                
                for cb in self.accumulate_metric_callbacks:
                    cb(pred, y_val)
                
                val_hist_batch.append(loss.item())
                pred_hist_batch.append(pred)
        val_loss = sum(val_hist_batch) / (len(dl_val) * dl_val.batch_size)     
        return val_loss

    def check_for_best_model(self, best_loss, curr_loss, iteration, idx_batch):
        if self.loss_lesser_is_better:
            is_improved = curr_loss < best_loss['value']
        else:
            is_improved = curr_loss > best_loss['value']
            
        if is_improved:
            best_loss['value'] = curr_loss
            best_loss['time'] = f'{self.iter_name}_{iteration}_batch_{idx_batch}'
            if self.save_only_best:
                model_save_name = f'best.pt'
            else:
                model_save_name = f'best_{self.iter_name}_{iteration}_batch_{idx_batch}.pt'
            torch.save(self.model, f'{self.save_dir}/{model_save_name}')
            self.early_stop_cnt = 0
        else:
            self.early_stop_cnt += 1
        return best_loss