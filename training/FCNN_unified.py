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
project_name = "MSU_interpol_unified_notebooks"

logger_path = './wandb_local_logs'
data_path = '../data/clasdb_pi_plus_n.txt'

hyperparams_dict = {
    'scale_data': False,
    'augment': True,
    'add_abc': False,
    'add_weights': False,
    'abc_loss_factor': None,
    'augment_factor': 25,
    'test_size': 0.1,
    'batch_size': 256,
    'net_architecture': [5, 60, 80, 100, 120, 140, 240, 340, 440, 640, 2000, 1040, 640, 340, 240, 140, 100, 80, 60, 20, 1],
    'activation_function': nn.ReLU(),
    'loss_func': 'RMSELoss()',
    'optim_func': torch.optim.Adam,
    'max_epochs': 2000,
    'es_min_delta': 0.00001,
    'es_patience': 20,
    'lr': 0.001,
    'lr_factor': 0.5,
    'lr_patience': 5,
    'lr_cooldown': 15,
}


# set up loss
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    @staticmethod
    def func_cos(x, a, b, c):
        return a + b * torch.cos(2 * x) + c * torch.cos(x)

    def forward(self, x, y_hat, y, w, A, B, C):
        if hyperparams_dict.get('add_abc') and hyperparams_dict.get('add_weights'):
            phi = x[:, 4]
            criterion = torch.sqrt(torch.mean(w * (y_hat - y) ** 2) / torch.sum(w)) + \
                        torch.mul(hyperparams_dict.get('abc_loss_factor'),
                                  torch.mean(torch.abs(w * y - self.func_cos(phi, A, B, C))) / torch.sum(w))
        elif hyperparams_dict.get('add_abc') and not hyperparams_dict.get('add_weights'):
            phi = x[:, 4]
            criterion = torch.sqrt(torch.mean((y_hat - y) ** 2)) + \
                        torch.mul(hyperparams_dict.get('abc_loss_factor'),
                                  torch.mean(torch.abs(y - self.func_cos(phi, A, B, C))))
        elif not hyperparams_dict.get('add_abc') and hyperparams_dict.get('add_weights'):
            criterion = torch.sqrt(torch.mean(w * (y_hat - y) ** 2) / torch.sum(w))
        else:
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
    def __init__(self, features, labels, weights, A, B, C):
        self.features = features
        self.labels = labels
        self.weights = weights
        self.A = A
        self.B = B
        self.C = C
        self.len = len(labels)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        weights = self.weights[index]
        A = self.A[index]
        B = self.B[index]
        C = self.C[index]
        return feature, label, weights, A, B, C

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
                          'dsigma_dOmega': np.random.normal(loc=new_augm.dsigma_dOmega, scale=new_augm.error / 3),
                          'error': new_augm.error,
                          'weight': new_augm.weight,
                          })
        if self.hyperparams.get('add_abc'):
            augm['A'] = new_augm.A
            augm['B'] = new_augm.B
            augm['C'] = new_augm.C
        else:
            pass
        return augm

    @staticmethod
    def func_cos(x, a, b, c):
        return a + b * np.cos(2 * x) + c * np.cos(x)

    def get_abc(self, df, E_beam, Q2, W, cos_theta):
        df_example_set = df[(df.Ebeam == E_beam) &
                            (df.W == W) &
                            (df.Q2 == Q2) &
                            (df.cos_theta == cos_theta)].sort_values('phi')
        # input data
        xdata = df_example_set.phi
        ydata = df_example_set.dsigma_dOmega
        ydata_error = df_example_set.error
        # fitting the data
        popt, pcov = curve_fit(self.func_cos, xdata, ydata, sigma=ydata_error, absolute_sigma=True)
        a, b, c = popt[0], popt[1], popt[2]

        return a, b, c

    def setup(self, stage):
        # data reading and preprocessing
        df = pd.read_csv(data_path, delimiter='\t', header=None)
        df.columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi', 'dsigma_dOmega', 'error', 'id']
        df.loc[8314:65671, 'Ebeam'] = 5.754  # peculiarity of this dataset.
        df = df[~((df.Ebeam == 5.754) & (~df.Q2.isin([1.715, 2.050, 2.445, 2.915, 3.480, 4.155])))] # peculiarity of this dataset #2
        df['phi'] = df.phi.apply(lambda x: math.radians(x))
        df['weight'] = df['error'].apply(lambda x: x and 1 / x or 100)  # x and 1 / x or 100  is just a reversed error but with validation 1/0 error in this case it will return 100
        df = df.drop('id', axis=1)
        df = df.drop_duplicates(subset=['Ebeam', 'W', 'Q2', 'cos_theta', 'phi'])

        # TODO: critical
        # Ebeam = [5.754]
        # Q2 = [1.72, 2.05, 2.44, 2.91, 3.48, 4.155]
        # df = df[(df.Q2.isin(Q2)) & (df.Ebeam.isin(Ebeam))]

        # Ebeam = [5.499]
        # W = [1.830, 1.890, 1.780, 1.950, 2.010, 1.620, 1.660, 1.700, 1.740]
        # df = df[df.Ebeam.isin(Ebeam) & (df.W.isin(W))]

        Ebeam = [1.515]
        df = df[df.Ebeam.isin(Ebeam)]

        # #train test split
        feature_columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']

        df['A'] = None
        df['B'] = None
        df['C'] = None
        feature_columns_with_additional = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi', 'weight', 'A', 'B', 'C']

        if self.hyperparams.get('add_abc'):
            for Ebeam in df.Ebeam.unique():
                for Q2 in tqdm.tqdm(df[df.Ebeam == Ebeam].Q2.unique(), desc='ABC Q cycle'):
                    for W in df[(df.Ebeam == Ebeam) & (df.Q2 == Q2)].W.unique():
                        for cos_theta in df[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W)].cos_theta.unique():
                            try:
                                if df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                        df.cos_theta == cos_theta), 'A'].iloc[0] is None:
                                    A, B, C = self.get_abc(df, Ebeam, Q2, W, cos_theta)
                                    df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                            df.cos_theta == cos_theta), 'A'] = A
                                    df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                            df.cos_theta == cos_theta), 'B'] = B
                                    df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                            df.cos_theta == cos_theta), 'C'] = C
                                else:
                                    pass
                            except Exception as e:
                                df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                        df.cos_theta == cos_theta), 'A'] = 0
                                df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                        df.cos_theta == cos_theta), 'B'] = 0
                                df.loc[(df.Ebeam == Ebeam) & (df.Q2 == Q2) & (df.W == W) & (
                                        df.cos_theta == cos_theta), 'C'] = 0
        else:
            pass

        feature_data = df[feature_columns_with_additional]
        label_data = df['dsigma_dOmega']

        if self.hyperparams.get('scale_data'):
            scaler_feature = StandardScaler()
            scaler_target = StandardScaler()
            feature_data = scaler_feature.fit_transform(feature_data)
            label_data = scaler_target.fit_transform(label_data.values.reshape(-1, 1))
        else:
            pass

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
            torch.tensor(train_label_data.values, dtype=torch.float32),
            torch.tensor(train_feature_data['weight'].values, dtype=torch.float32),
            torch.tensor(train_feature_data['A'].astype(float).values, dtype=torch.float32),
            torch.tensor(train_feature_data['B'].astype(float).values, dtype=torch.float32),
            torch.tensor(train_feature_data['C'].astype(float).values, dtype=torch.float32))

        self.val_dataset = InterpolDataSet(torch.tensor(val_feature_data[feature_columns].values, dtype=torch.float32),
                                           torch.tensor(val_label_data.values, dtype=torch.float32),
                                           torch.tensor(val_feature_data['weight'].values, dtype=torch.float32),
                                           torch.tensor(train_feature_data['A'].astype(float).values,
                                                        dtype=torch.float32),
                                           torch.tensor(train_feature_data['B'].astype(float).values,
                                                        dtype=torch.float32),
                                           torch.tensor(train_feature_data['C'].astype(float).values,
                                                        dtype=torch.float32))

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
        x, y, w, A, B, C = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.train_loss = loss.forward(x=x, y_hat=y_hat.reshape(-1), y=y, w=w, A=A, B=B, C=C)
        self.train_mae = self.mae(y_hat.reshape(-1), y)

        self.log('train_loss', self.train_loss, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('train_mae', self.train_mae, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.training_step_outputs.append(self.train_loss)
        return self.train_loss

    def validation_step(self, batch, batch_idx):
        x, y, w, A, B, C = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.val_loss = loss.forward(x=x, y_hat=y_hat.reshape(-1), y=y, w=w, A=A, B=B, C=C)
        self.val_mae = self.mae(y_hat.reshape(-1), y)

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
