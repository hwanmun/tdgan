""" Script to create synthetic tabular data using a generator model. """

import sys
import yaml
import numpy as np
import torch
from dataset import SalesDataset
from model import FCGenerator
from util import vec_to_table


if __name__ == '__main__':
    model_path, config_path, table_size, csv_path = sys.argv[1:5]
    table_size = int(table_size)
    # training config file needed to recreate the generator model.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ds_train = SalesDataset(
        config['train_data_path'], store_fmt=config['store_fmt'],
        dayofweek_fmt=config['dayofweek_fmt'], date_fmt=config['date_fmt'],
        noise_size=config['noise_size']
    )
    feat_dim = ds_train.feat_dim

    state = torch.load(model_path)
    gen = FCGenerator(config['latent_dim'], feat_dim, config['g_hdims'])
    gen.load_state_dict(state['state_dict'])

    gen.eval()
    data = np.array([])
    batch_size = 512
    while data.shape[0] < table_size:
        b_size = min(table_size - data.shape[0], batch_size)
        z = torch.randn(b_size, config['latent_dim'], device='cpu')
        vec = gen(z).detach().numpy()
        if data.shape[0] == 0:
            data = vec
        else:
            data = np.concatenate((data, vec), axis=0)
    
    df = vec_to_table(data, ds_train.key_to_idx, ds_train.stats)
    df.to_csv(csv_path, index=False)