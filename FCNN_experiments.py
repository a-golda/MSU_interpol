import os
import sys
import math
import wandb
import torch
import logging

import pandas as pd
import torch.nn as nn
import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torchmetrics import MeanAbsoluteError, MeanSquaredError

# paths
logger_path = './wandb_local_logs'
data_path = './data/clasdb_pi_plus_n.txt'

# wandb and logs name
new_experiment_name = f"baseline"
project_name = "MSU_interpol_experiments"

# set up wandb
wandb.login()
wandb_logger = WandbLogger(project=project_name,
                           save_dir=logger_path)
initial_exp_name = wandb_logger.experiment.name
wandb_logger.experiment.name = new_experiment_name
wandb.run.name = new_experiment_name

# set up loggers
logger_full_path = os.path.join(logger_path, project_name, new_experiment_name)

os.makedirs(logger_full_path, exist_ok=True)
logging.basicConfig(encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.FileHandler(os.path.join(logger_full_path, 'logs.log'), mode='w'),
                              logging.StreamHandler(sys.stdout)],
                    force=True)

#
hyperparams = {
    'test_size': 0.1,
    'batch_size': 256,
    'random_state': 1438,
    'loss_function': MeanSquaredError(),
    'net_architecture': [5, 60, 80, 100, 120, 140, 240, 340, 440, 640, 2000, 1040, 640, 340, 240, 140, 100, 80, 60, 20, 1],
    'activation_function': nn.ReLU(),
    'optim_func': torch.optim.Adam,
    'max_epochs': 2000,
    'es_min_delta': 0.00001,
    'es_patience': 20,
    'lr': 0.001,
    'lr_factor': 0.5,
    'lr_patience': 5,
    'lr_cooldown': 15
}


# define dataset
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


# define net
class InterpolDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage):
        # data reading and preprocessing
        df = pd.read_csv(data_path, delimiter='\t', header=None)
        df.columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi', 'dsigma_dOmega', 'error', 'id']
        df.loc[8314:65671, 'Ebeam'] = 5.754  # peculiarity of this dataset.
        df = df[~((df.Ebeam == 5.754) &
                  (~df.Q2.isin([1.715, 2.050, 2.445, 2.915, 3.480, 4.155])))]  # peculiarity of this dataset #2
        df['phi'] = df.phi.apply(lambda x: math.radians(x))
        df = df.drop('id', axis=1)
        df = df.drop_duplicates(subset=['Ebeam', 'W', 'Q2', 'cos_theta', 'phi'])

        # train test split
        feature_columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']
        label_column = 'dsigma_dOmega'

        feature_data = df[feature_columns]
        label_data = df[label_column]

        (train_feature_data, val_feature_data,
         train_label_data, val_label_data) = train_test_split(feature_data,
                                                              label_data,
                                                              test_size=hyperparams.get('test_size'),
                                                              random_state=hyperparams.get('random_state'))

        self.train_dataset = InterpolDataSet(
            features=torch.tensor(train_feature_data[feature_columns].values, dtype=torch.float32),
            labels=torch.tensor(train_label_data.values, dtype=torch.float32))

        self.val_dataset = InterpolDataSet(
            features=torch.tensor(val_feature_data[feature_columns].values, dtype=torch.float32),
            labels=torch.tensor(val_label_data.values, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=hyperparams.get('batch_size'),
                          shuffle=False,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=hyperparams.get('batch_size'),
                          shuffle=False,
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
    def __init__(self):
        super(InterpolRegressor, self).__init__()

        self.save_hyperparameters(hyperparams)

        (self.train_loss, self.train_mae, self.train_mse,
         self.val_loss, self.val_mae, self.val_mse) = 0, 0, 0, 0, 0, 0

        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.loss_func = hyperparams.get('loss_function')

        self.optim = hyperparams.get('optim_func')

        self.net_architecture = hyperparams.get('net_architecture')
        self.activation_function = hyperparams.get('activation_function')

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
        self.train_loss = loss.forward(target=y, preds=y_hat.reshape(-1))
        self.train_mae = self.mae(y_hat.reshape(-1), y)
        self.train_mse = self.mse(y_hat.reshape(-1), y)

        self.log('train_loss', self.train_loss, batch_size=hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('train_mae', self.train_mae, batch_size=hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('train_mse', self.train_mse, batch_size=hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.training_step_outputs.append(self.train_loss)
        return self.train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.val_loss = loss.forward(target=y, preds=y_hat.reshape(-1))
        self.val_mae = self.mae(y_hat.reshape(-1), y)
        self.val_mse = self.mse(y_hat.reshape(-1), y)

        self.log('val_loss', self.val_loss, batch_size=hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('val_mae', self.val_mae, batch_size=hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('val_mse', self.val_mse, batch_size=hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.validation_step_outputs.append(self.val_loss)
        return self.val_loss

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.trainer.current_epoch != 0:
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            pass

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min",
                                            min_delta=hyperparams.get('es_min_delta'),
                                            patience=hyperparams.get('es_patience'),
                                            verbose=True)

        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              monitor="val_loss",
                                              mode="min",
                                              dirpath=f"{logger_full_path}/checkpoints",
                                              filename="{exp_name}{val_loss:.5f}-{epoch:02d}")

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        print_callback = PrintCallbacks()

        return [early_stop_callback, checkpoint_callback, print_callback, lr_monitor]

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(),
                               lr=hyperparams.get('lr'))
        lr_optim = ReduceLROnPlateau(optimizer=optimizer,
                                     mode='min',
                                     factor=hyperparams.get('lr_factor'),
                                     patience=hyperparams.get('lr_patience'),
                                     cooldown=hyperparams.get('lr_cooldown'),
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


data_module = InterpolDataModule()
model = InterpolRegressor()
trainer = pl.Trainer(max_epochs=hyperparams.get('max_epochs'),
                     accelerator='gpu',
                     logger=wandb_logger,
                     enable_progress_bar=False)
trainer.fit(model, data_module)

wandb.finish()
