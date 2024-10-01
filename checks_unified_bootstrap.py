import os
import sys

import tqdm
import math
import torch
import logging

import numpy as np
import pandas as pd
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from scipy import spatial
from scipy.stats import chisquare, kstest
from scipy.optimize import curve_fit
from torchmetrics import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint


hyperparams_dict = {
        'energy': 1.515,
        'scale_data': False,
        'augment': False,
        'add_abc': False,
        'abc_loss_factor': 1,
        'augment_factor': 20,
        'test_size': 0.1,
        'batch_size': 256,
        'net_architecture': [5,60,80,100,120,140,240,340,440,640,2000,1040,640,340,240,140,100,80,60,20,1],
        'activation_function': nn.ReLU(),
        'loss_func': 'RMSELoss()',
        'optim_func': torch.optim.Adam,
        'max_epochs': 2000,
        'es_min_delta': 0.00001,
        'es_patience': 50,
        'lr': 0.001,
        'lr_factor':0.5,
        'lr_patience': 5,
        'lr_cooldown': 20,
    }


class RMSELoss(torch.nn.Module):
    def __init__(self, add_abc=False):
        super(RMSELoss, self).__init__()
        self.add_abc = add_abc

    @staticmethod
    def func_cos(x, a, b, c):
        return a + b * torch.cos(2 * x) + c * torch.cos(x)

    def forward(self, x, y_hat, y, w, A, B, C):
        if self.add_abc:
            phi = x[:, 4]
            criterion = torch.sqrt(torch.mean(w * (y_hat - y) ** 2) / torch.sum(w)) + \
                        torch.mul(hyperparams_dict.get('abc_loss_factor'),
                                  torch.mean(torch.abs(w * y - self.func_cos(phi, A, B, C))) / torch.sum(w))
        else:
            criterion = torch.sqrt(torch.mean(w * (y_hat - y) ** 2) / torch.sum(w))
        return criterion


# params
project_name = "MSU_interpol_unified_notebooks"

logger_path = './wandb_local_logs'
data_path = './data/clasdb_pi_plus_n.txt'

logger_full_path = os.path.join(logger_path, project_name, 'spring-feather-42')

os.makedirs(logger_full_path, exist_ok=True)
logging.basicConfig(encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.FileHandler(os.path.join(logger_full_path, 'logs.log'), mode='w'),
                              logging.StreamHandler(sys.stdout)],
                    force=True)


class InterpolDataSet(Dataset):
    def __init__(self, features, labels, weights):
        self.features = features
        self.labels = labels
        self.weights = weights
        self.len = len(labels)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        weights = self.weights[index]
        return feature, label, weights

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
        return augm

    def setup(self, stage):
        # data reading and preprocessing
        df = pd.read_csv(data_path, delimiter='\t', header=None)
        df.columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi', 'dsigma_dOmega', 'error', 'id']
        df.loc[8314:65671, 'Ebeam'] = 5.754  # peculiarity of this dataset.
        df = df[~((df.Ebeam == 5.754) & (
            ~df.Q2.isin([1.715, 2.050, 2.445, 2.915, 3.480, 4.155])))]  # peculiarity of this dataset #2
        df['phi'] = df.phi.apply(lambda x: math.radians(x))
        df['weight'] = df['error'].apply(lambda
                                             x: x and 1 / x or 100)  # x and 1 / x or 100  is just a reversed error but with validation 1/0 error in this case it will return 100
        df = df.drop('id', axis=1)
        df = df.drop_duplicates(subset=['Ebeam', 'W', 'Q2', 'cos_theta', 'phi'])

        df = df[df.Ebeam == hyperparams_dict.get('energy')]

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
            torch.tensor(train_feature_data['weight'].values, dtype=torch.float32))

        self.val_dataset = InterpolDataSet(torch.tensor(val_feature_data[feature_columns].values, dtype=torch.float32),
                                           torch.tensor(val_label_data.values, dtype=torch.float32),
                                           torch.tensor(val_feature_data['weight'].values, dtype=torch.float32))

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
        self.loss_func = self.hyperparams.get('loss_func')

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
        x, y, w = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.train_loss = loss.forward(y_hat.reshape(-1), y, w)
        self.train_mae = self.mae(y_hat.reshape(-1), y)

        self.log('train_loss', self.train_loss, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.log('train_mae', self.train_mae, batch_size=self.hyperparams['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        self.training_step_outputs.append(self.train_loss)
        return self.train_loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self.forward(x)

        loss = self.loss_func
        self.val_loss = loss.forward(y_hat.reshape(-1), y, w)
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


data = InterpolDataModule(hyperparams_dict)
data.setup('test')
df = data.df
df_all = df[['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']]
df_target = df[['dsigma_dOmega']]

# generate grid
if True:
    Ebeam = hyperparams_dict.get('energy')
    step_W = 0.005
    step_Q2 = 0.1
    step_cos_theta = 0.1
    step_phi = 0.05

    W_min = df[df.Ebeam==Ebeam].W.min() - 0.1
    W_max = df[df.Ebeam==Ebeam].W.max() + 0.1 + step_W

    Q2_min = df[df.Ebeam==Ebeam].Q2.min() - 0.1
    Q2_max = df[df.Ebeam==Ebeam].Q2.max() + 0.1 + step_Q2

    data_grid = []
    for W in tqdm.tqdm(np.arange(W_min, W_max, step_W)):
        for Q2 in np.arange(Q2_min, Q2_max, step_Q2):
             for cos_theta in np.arange(-1, 1+step_cos_theta, step_cos_theta):
                    for phi in np.arange(0, 2*np.pi, step_phi):
                        data_grid.append([Ebeam,W,Q2,cos_theta,phi])

    df_grid = pd.DataFrame(data_grid)
    df_grid.columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']

    df_grid.W = np.round(df_grid.W, 3)
    df_grid.Q2 = np.round(df_grid.Q2, 3)
    df_grid.cos_theta = np.round(df_grid.cos_theta, 3)
    df_grid.phi = np.round(df_grid.phi, 3)

df_grid_base = df_grid.copy()

data = []
all_models = os.listdir('./legacy_notebooks/training/wandb_local_logs/MSU_interpol_bootstrap/')
for model in all_models:
    data.append([int(model.split('-')[-1]), model, os.listdir(f'./legacy_notebooks/training/wandb_local_logs/MSU_interpol_bootstrap/{model}/checkpoints/')[0]])

data_en = ['./data/models/wandering-wind-26/checkpoints/exp_name=0val_loss=1.38199-epoch=233.ckpt']
for data_entry in data:
    if data_entry[0] <= 62 and data_entry[0] >= 43:
        data_en.append([data_entry[0], f'./legacy_notebooks/training/wandb_local_logs/MSU_interpol_bootstrap/{data_entry[1]}/checkpoints/{data_entry[2]}'])

df_grid_global = pd.DataFrame()

for model_path in data_en:
    print(model_path)

    df_grid = df_grid_base.copy()

    try:
        model = InterpolRegressor.load_from_checkpoint(model_path[1], hyperparams=hyperparams_dict)
        model.eval()

        # predict crosssections
        if True:
            df_grid_parts = np.array_split(df_grid, 100)
            df_grid_parts_preds = []
            for df_grid_part in tqdm.tqdm(df_grid_parts):
                dsigma_dOmega_predicted = model.forward(torch.tensor(df_grid_part.to_numpy(),dtype=torch.float32)).detach()

                df_grid_part['dsigma_dOmega_predicted'] = dsigma_dOmega_predicted
                df_grid_part.dsigma_dOmega_predicted = abs(df_grid_part.dsigma_dOmega_predicted)

                df_grid_part['A'] = 0
                df_grid_part['B'] = 0
                df_grid_part['C'] = 0
                df_grid_parts_preds.append(df_grid_part)

            df_grid = pd.concat(df_grid_parts_preds)

        # calculate structure functions
        if True:
            phi_min_index = df_grid[df_grid.phi == df_grid.phi.min()].index.to_numpy()
            phi_max_index = df_grid[df_grid.phi == df_grid.phi.max()].index.to_numpy()

            for i in tqdm.tqdm(range(len(phi_min_index))):
                cross_section_chunk = df_grid.iloc[phi_min_index[i]:phi_max_index[i]].dsigma_dOmega_predicted
                cos_phi = np.cos(df_grid.iloc[phi_min_index[i]:phi_max_index[i]].phi)
                cos_2_phi = np.cos(2*df_grid.iloc[phi_min_index[i]:phi_max_index[i]].phi)

                trapz_A = np.trapz(cross_section_chunk, dx=step_phi)
                trapz_B = np.trapz(cross_section_chunk*cos_2_phi, dx=step_phi)
                trapz_C = np.trapz(cross_section_chunk*cos_phi, dx=step_phi)

                A = trapz_A/(2*np.pi)
                B = trapz_B/(np.pi)
                C = trapz_C/(np.pi)

                df_grid.loc[phi_min_index[i]:phi_max_index[i], 'A'] = A
                df_grid.loc[phi_min_index[i]:phi_max_index[i], 'B'] = B
                df_grid.loc[phi_min_index[i]:phi_max_index[i], 'C'] = C

        df_grid['dsigma_dOmega_sf'] = df_grid['A'] + df_grid['B'] * np.cos(2 * df_grid['phi']) + df_grid['C'] * np.cos(
            df_grid['phi'])

        if len(df_grid_global) == 0:
            df_grid_global = df_grid
        else:
            df_grid_global = pd.merge(df_grid_global, df_grid, on=['Ebeam', 'W', 'Q2', 'cos_theta', 'phi'], how='inner', suffixes=('', f'_{model_path[0]}'))
    except Exception as e:
        print(e)

df_grid_global.to_csv('./data/bootstrap_csv_1515.csv')