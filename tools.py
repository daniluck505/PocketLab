import yaml
import os
from datetime import datetime
import torch

def load_options(path):
    with open(path, 'r') as f:
        options = yaml.safe_load(f)
    return options

def download_kaggle_data(name):
    if 'kaggle.json' not in os.listdir():
        raise Exception(f'kaggle.json not found')
    os.system('pip install kaggle')
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system(f'kaggle datasets download "{name}"')
    os.system(f'unzip "{name.split("/")[-1]}.zip"')
    
    
def save_results(options, model, loss_history, acc_history, test_history):
    if not os.path.exists(f'{options["name"]}'):
        os.system(f'mkdir {options["name"]}')
    now = datetime.now()
    now = str(now).split('.')[0].replace(' ','_' )
    name = f'{options["network"]["arch"]}_{now}'
    torch.save(model.state_dict(), f'{options["name"]}/{name}_weights.pt') 

    with open(f'{options["name"]}/{name}_results.txt', 'w') as f:
        f.writelines(f'loss_history: {str(loss_history)}')
        f.writelines(f'acc_history: {str(acc_history)}')
        f.writelines(f'test_history: {str(test_history)}')
        f.writelines(str(options))