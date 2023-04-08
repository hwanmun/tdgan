""" Script to train GAN model. Model/training hyperparameters and
relevant paths should be stored in the training config file.
"""

import os
import sys
import yaml
from dataset import SalesDataset
from model import FCGenerator, FCDiscriminator
from interface import TrainingInterfaceGAN

if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    exp_name = config_path.split('/')[-1][:-5]

    ds_train = SalesDataset(
        config['train_data_path'], store_fmt=config['store_fmt'],
        dayofweek_fmt=config['dayofweek_fmt'], date_fmt=config['date_fmt'],
        noise_size=config['noise_size']
    )
    ds_dev = SalesDataset(
        config['dev_data_path'], store_fmt=config['store_fmt'],
        dayofweek_fmt=config['dayofweek_fmt'], date_fmt=config['date_fmt'],
        stats_from=ds_train
    )

    zdim, fdim = config['latent_dim'], ds_train.feat_dim

    gen = FCGenerator(zdim, fdim, config['g_hdims'])
    disc = FCDiscriminator(fdim, config['d_hdims'])
    log_path = os.path.join(config['log_dir'], f'{exp_name}.log')
    model_path = os.path.join(config['model_dir'], f'{exp_name}.pt')
    eval_model = config['eval_model'] if 'eval_model' in config else 'lin_reg'
    interface = TrainingInterfaceGAN(gen, disc, ds_train, ds_dev, zdim,
                                     model_path=model_path, log_path=log_path,
                                     eval_kwargs={'eval_model': eval_model})

    g_betas = (config['g_beta1'], config['g_beta2'])
    d_betas = (config['d_beta1'], config['d_beta2'])
    gen_kwargs = {'lr': config['g_lr'], 'betas': g_betas}
    disc_kwargs = {'lr': config['d_lr'], 'betas': d_betas}
    interface.train(epochs=config['epochs'], batch_size=config['batch_size'],
                    gen_opt_kwargs=gen_kwargs, disc_opt_kwargs=disc_kwargs,
                    running_avg_size=config['running_avg_size'],
                    train_tracking_period=config['train_tracking_period'])