""" Script to evaluate a generator model. """

import argparse
import os
import yaml
import torch
from dataset import SalesDataset
from model import FCGenerator
from evaluation import EvaluatorByTraining

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True,
                        help='Path of the data file.')
    parser.add_argument('-m', '--model', required=True,
                        help='Path of the model file.')
    parser.add_argument('-c', '--config', required=True,
                        help='Path of the config file used to train the model.')
    parser.add_argument('-k', '--labelkey', default='Sales',
                        help='Key to use as the label.')
    parser.add_argument('-e', '--evalmodel', default='lin_reg',
                        help='Model to train for evaluation.')
    parser.add_argument('-t', '--metric', default='rmspe',
                        help='Metric to evaluate model performace.')
    parser.add_argument('-o', '--comparison', default='ratio',
                        help='Method to compare the fake and real metrics.')
    parser.add_argument('-l', '--log_dir', default='evals',
                        help='Directory to store evaluation log.')
    args = parser.parse_args()

    # training config file needed to recreate the generator model.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    exp_name = args.config.split('/')[-1][:-5]
    
    ds_train = SalesDataset(
        config['train_data_path'], store_fmt=config['store_fmt'],
        dayofweek_fmt=config['dayofweek_fmt'], date_fmt=config['date_fmt'],
        noise_size=config['noise_size']
    )
    feat_dim = ds_train.feat_dim

    ds_test = SalesDataset(
        args.data, store_fmt=config['store_fmt'],
        dayofweek_fmt=config['dayofweek_fmt'], date_fmt=config['date_fmt'],
        noise_size=config['noise_size'], stats_from=ds_train, sales_only=True
    )

    state = torch.load(args.model)
    gen = FCGenerator(config['latent_dim'], feat_dim, config['g_hdims'])
    gen.load_state_dict(state['state_dict'])

    evaluator = EvaluatorByTraining(ds_test, eval_model=args.evalmodel,
                                    metric=args.metric,
                                    comparison=args.comparison)
    
    value, (fake, real) = evaluator.evaluate(gen, config['latent_dim'])

    result = f'Evaluated model at {args.model} with dataset at {args.data}. \n'
    result += f'[eval_{args.comparison}={value}, fake_{args.metric}={fake}, '
    result += f'real_{args.metric}={real}]'


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, f'eval_{exp_name}'), 'w') as f:
        f.write(result)
    print(result)