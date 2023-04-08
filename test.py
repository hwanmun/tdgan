""" Unit tests """

import os
import numpy as np
import torch
from dataset import split_sales_dataset, SalesDataset
from model import FCGenerator, FCDiscriminator
from losses import gen_BCE_logit_loss, disc_BCE_logit_loss
from util import vec_to_table
from evaluation import EvaluatorByTraining
from interface import TrainingInterfaceGAN

def test_dataset():
    split_sales_dataset('data/store.csv', 'data/train_tiny.csv', 'ds_test',
                        'tiny', split=[0.9, 0.05, 0.05], random_seed=2)
    assert os.path.exists('ds_test/tiny_train.csv')
    assert os.path.exists('ds_test/tiny_dev.csv')
    assert os.path.exists('ds_test/tiny_test.csv')

    file_path = 'ds_test/tiny_dev.csv'
    datasets = []
    datasets.append((SalesDataset(file_path), 1141))
    datasets.append((SalesDataset(file_path, store_fmt='numeral'), 27))
    datasets.append((SalesDataset(file_path, dayofweek_fmt='onehot'), 1146))
    datasets.append((SalesDataset(file_path, date_fmt='numeral'), 1139))
    datasets.append((SalesDataset(file_path, sales_only=True), 1128))
    for ds, dim in datasets:
        for idx, batch in enumerate(ds.get_dataloader(batch_size=16)):
            if idx > 1:
                break
            assert batch.shape == (16, dim)
    
    ds = SalesDataset(file_path)
    dl = ds.get_dataloader(batch_size=128, drop_last=False)
    for batch in dl:
        assert not np.any(np.isnan(batch.numpy()))
    
   
def test_model():
    gen = FCGenerator(64, 1024, [2048, 2048])
    disc = FCDiscriminator(1024, [2048])
    with torch.no_grad():
        x = torch.randn(16, 64)
        x = gen(x)
        assert tuple(x.shape) == (16, 1024)
        x = disc(x)
        assert tuple(x.shape) == (16, 1)

def test_losses():
    real = torch.nn.functional.sigmoid(torch.randn(16, 1))
    fake = torch.nn.functional.sigmoid(torch.randn(16, 1))
    gen_loss = gen_BCE_logit_loss(fake).item()
    disc_loss = disc_BCE_logit_loss(real, fake).item()

    real, fake = real.numpy(), fake.numpy()
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    gen_loss_np = -np.log(sig(fake)).mean()
    disc_loss_np = -(np.log(sig(real)) + np.log(1 - sig(fake))).mean() / 2
    
    assert np.allclose([gen_loss, disc_loss], [gen_loss_np, disc_loss_np])

def test_util():
    assert os.path.exists('ds_test/tiny_test.csv')
    ds = SalesDataset('ds_test/tiny_test.csv')
    ds.data = ds.data.sample(frac=1).reset_index()
    batch_size = 12
    dl = ds.get_dataloader(batch_size=batch_size, shuffle=False)
    for batch in dl:
        break
    batch = batch.numpy()

    sales_keys = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open',
                  'Promo', 'StateHoliday', 'SchoolHoliday']
    df_org = ds.data[sales_keys][:batch_size]
    df = vec_to_table(batch, ds.key_to_idx, ds.stats)
    
    assert df.equals(df_org)

def test_evaluation():
    ds = SalesDataset('ds_test/tiny_dev.csv', sales_only=True)
    evaluator = EvaluatorByTraining(ds, eval_model='dec_tree_reg')

    state = torch.load('models/tiny.pt')
    gen = FCGenerator(32, 1141, [128])
    gen.load_state_dict(state['state_dict'])

    prod, repeat = 1, 5
    for _ in range(repeat):
        ratio, (fake_metric, real_metric) = evaluator.evaluate(gen, 32)
        prod *= ratio
    assert prod > 1.0

def test_interface():
    assert os.path.exists('ds_test/tiny_train.csv')
    assert os.path.exists('ds_test/tiny_dev.csv')
    assert os.path.exists('ds_test/tiny_test.csv')
    
    latent_dim, hdim, fdim = 32, 128, 1141
    
    ds_train = SalesDataset('ds_test/tiny_train.csv', noise_size=0.05)
    ds_dev = SalesDataset('ds_test/tiny_dev.csv', stats_from=ds_train,
                          sales_only=True)

    gen = FCGenerator(latent_dim, fdim, [hdim, hdim])
    disc = FCDiscriminator(fdim, [hdim, hdim])
    interface = TrainingInterfaceGAN(gen, disc, ds_train, ds_dev, latent_dim,
                                     log_path='ds_test/test.log',
                                     eval_kwargs={'eval_model': 'lin_reg'})

    gen_kwargs = {'lr': 1E-4, 'betas': (0.5, 0.999)}
    disc_kwargs = {'lr': 3E-5, 'betas': (0.5, 0.999)}
    interface.train(epochs=5, batch_size=512,
                    gen_opt_kwargs=gen_kwargs, disc_opt_kwargs=disc_kwargs,
                    running_avg_size=10, train_tracking_period=50)


