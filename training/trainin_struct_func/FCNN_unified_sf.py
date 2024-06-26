import os
import sys
import math
import tqdm
import wandb
import torch
import logging

import numpy as np
import pandas as pd
import torch.nn as nn
import lightning.pytorch as pl

from scipy.optimize import curve_fit
from torchmetrics import MeanAbsoluteError
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

wandb.login()

# params
project_name = "MSU_interpol_unified_notebooks_structure_functions"

logger_path = '../wandb_local_logs'
data_path = '../../data/df_with_ABC.csv'

hyperparams_dict = {
    'scale_data': False,
    'augment': False,
    'add_weights': False,
    'augment_factor': 20,
    'test_size': 0.1,
    'batch_size': 256,
    'net_architecture': [5, 60, 80, 100, 120, 140, 240, 340, 440, 640, 2000, 1040, 640, 340, 240, 140, 100, 80, 60, 20, 3],
    'activation_function': nn.ReLU(),
    'loss_func': 'RMSELoss()',
    'optim_func': torch.optim.Adam,
    'max_epochs': 2000,
    'es_min_delta': 0.00001,
    'es_patience': 50,
    'lr': 0.001,
    'lr_factor': 0.5,
    'lr_patience': 5,
    'lr_cooldown': 20,
}


# set up loss
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_hat, y):
        criterion = torch.sqrt(torch.mean((y_hat - y)** 2))
        return criterion

global_losss_function = RMSELoss()

# set up loggers and wandb
wandb_logger = WandbLogger(project=project_name,
                           save_dir=logger_path)
exp_name = wandb_logger.experiment.name

logger_full_path = os.path.join(logger_path, project_name, exp_name)

os.makedirs(logger_full_path, exist_ok=True)
logging.basicConfig(encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.FileHandler(os.path.join(logger_full_path, 'logs.log'), mode='w'),
                              logging.StreamHandler(sys.stdout)],
                    force=True)


# define dataset and net
class InterpolDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.len = len(labels)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return self.len


class InterpolDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.df = None
        self.hyperparams = hyperparams
        self.train_dataset = None
        self.val_dataset = None

    def augment(self, new_augm):
        augm = pd.Series({'Ebeam': np.random.normal(loc=new_augm.Ebeam, scale=new_augm.Ebeam / 30),
                          'W': np.random.normal(loc=new_augm.W, scale=new_augm.W / 30),
                          'Q2': np.random.normal(loc=new_augm.Q2, scale=new_augm.Q2 / 30),
                          'cos_theta': np.clip(
                              np.random.normal(loc=new_augm.cos_theta, scale=abs(new_augm.cos_theta / 30)), -1, 1),
                          'phi': np.clip(np.random.normal(loc=new_augm.phi, scale=new_augm.phi / 30), 0, 2 * np.pi),
                          'A': np.random.normal(loc=new_augm.A, scale=new_augm.A_error / 3),
                          'B': np.random.normal(loc=new_augm.B, scale=new_augm.B_error / 3),
                          'C': np.random.normal(loc=new_augm.C, scale=new_augm.C_error / 3)
                          })
        return augm

    def setup(self, stage):
        # data reading and preprocessing
        df = pd.read_csv(data_path)
        df = df.dropna()

        feature_columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']

        feature_data = df[feature_columns]
        label_data = df[['A', 'B', 'C']]

        if self.hyperparams.get('augment'):
            aug_series_list = []
            for i in tqdm.tqdm(df.itertuples()):
                for _ in range(self.hyperparams.get('augment_factor')):
                    aug_series_list.append(self.augment(i))

            aug_df = pd.DataFrame(aug_series_list)
            df = pd.concat([df, aug_df])
        else:
            pass

        self.df = df

        train_feature_data, val_feature_data, train_label_data, val_label_data = train_test_split(feature_data,
                                                                                                  label_data,
                                                                                                  test_size=self.hyperparams.get(
                                                                                                      'test_size'),
                                                                                                  random_state=1438)

        self.train_dataset = InterpolDataSet(
            torch.tensor(train_feature_data[feature_columns].values, dtype=torch.float32),
            torch.tensor(train_label_data.values, dtype=torch.float32))

        self.val_dataset = InterpolDataSet(torch.tensor(val_feature_data[feature_columns].values, dtype=torch.float32),
                                           torch.tensor(val_label_data.values, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.hyperparams.get('batch_size'), shuffle=False,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.hyperparams.get('batch_size'), shuffle=False,
                          num_workers=0)


class PrintCallbacks(Callback):
    def on_train_start(self, trainer, pl_module):
        logging.info("Training is starting")

    def on_train_end(self, trainer, pl_module):
        logging.info("Training is ending")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
        logging.info(f"epoch: {pl_module.current_epoch}; train_loss: {epoch_mean}")
        pl_module.training_step_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_mean = torch.stack(pl_module.validation_step_outputs).mean()
        logging.info(f"epoch: {pl_module.current_epoch}; val_loss: {epoch_mean}")
        pl_module.validation_step_outputs.clear()


class InterpolRegressor(pl.LightningModule):
    def __init__(self, hyperparams):
        super(InterpolRegressor, self).__init__()

        self.train_loss, self.train_mae, self.val_loss, self.val_mae = 0, 0, 0, 0
        self.hyperparams = hyperparams
        self.save_hyperparameters(self.hyperparams)

        self.mae = MeanAbsoluteError()
        self.loss_func = global_losss_function

        self.optim = self.hyperparams.get('optim_func')

        self.net_architecture = self.hyperparams.get('net_architecture')
        self.activation_function = self.hyperparams.get('activation_function')

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.net = nn.Sequential()
        for i in range(1, len(self.net_architecture)):
            self.net.append(nn.Linear(self.net_architecture[i - 1], self.net_architecture[i]))
            if i != len(self.net_architecture) - 1:
                self.net.append(self.activation_function)
            else:
                pass

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.train_loss = loss.forward(y_hat=y_hat.reshape(-1), y=y.reshape(-1))
        self.train_mae = self.mae(y_hat.reshape(-1), y.reshape(-1))

        self.log('train_loss', self.train_loss, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('train_mae', self.train_mae, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.training_step_outputs.append(self.train_loss)
        return self.train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.val_loss = loss.forward(y_hat=y_hat.reshape(-1), y=y.reshape(-1))
        self.val_mae = self.mae(y_hat.reshape(-1), y.reshape(-1))

        self.log('val_loss', self.val_loss, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('val_mae', self.val_mae, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.validation_step_outputs.append(self.val_loss)
        return self.val_loss

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.trainer.current_epoch != 0:
            sch.step(self.trainer.callback_metrics["val_loss"])

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min",
                                            min_delta=self.hyperparams.get('es_min_delta'),
                                            patience=self.hyperparams.get('es_patience'),
                                            verbose=True)

        checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                              monitor="val_loss",
                                              mode="min",
                                              dirpath=f"{logger_full_path}/checkpoints",
                                              filename="{exp_name}{val_loss:.5f}-{epoch:02d}")

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        print_callback = PrintCallbacks()

        return [early_stop_callback, checkpoint_callback, print_callback, lr_monitor]

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.hyperparams.get('lr'))
        lr_optim = ReduceLROnPlateau(optimizer=optimizer,
                                     mode='min',
                                     factor=self.hyperparams.get('lr_factor'),
                                     patience=self.hyperparams.get('lr_patience'),
                                     cooldown=self.hyperparams.get('lr_cooldown'),
                                     threshold=0.01,
                                     verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_optim,
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 2,
                    "name": 'lr_scheduler_monitoring'}
                }


data_module = InterpolDataModule(hyperparams=hyperparams_dict)
model = InterpolRegressor(hyperparams=hyperparams_dict)
trainer = pl.Trainer(max_epochs=hyperparams_dict.get('max_epochs'),
                     accelerator='cpu',
                     logger=wandb_logger,
                     enable_progress_bar=False)
trainer.fit(model, data_module)

wandb.finish()
