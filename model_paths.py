import os

data = []
all_models = os.listdir('./legacy_notebooks/training/wandb_local_logs/MSU_interpol_bootstrap/')
for model in all_models:
    data.append([int(model.split('-')[-1]), model, os.listdir(f'./legacy_notebooks/training/wandb_local_logs/MSU_interpol_bootstrap/{model}/checkpoints/')[0]])

print(data)
