import copy
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from util import regularize_vec

class EvaluatorByTraining:
    """ Evaluate the performance of a generator model.
    For that, further split the test dataset into 'train' and 'test' set,
    and create 'fake train' data using the generator model. Train two
    separate models within the tabular data (ex: perdicting Sales) and
    measure the peformance of two models using certain metric (ex:RMSPE).
    Also present the ratio/margin of two metrics.
    """
    def __init__(self, dataset, label_key='Sales', eval_model='lin_reg',
                 metric='rmspe', comparison='ratio', train_frac=0.8,
                 batch_size=512, device='cpu'):
        ds_train = copy.deepcopy(dataset)
        ds_test = copy.deepcopy(dataset)
        self.n_train = round(len(ds_train) * train_frac)
        ds_train.data = ds_train.data.sample(frac=1).reset_index()
        ds_test.data = ds_train.data[self.n_train:].reset_index()
        ds_train.data = ds_train.data[:self.n_train].reset_index()
        self.key_to_idx = dict(ds_train.key_to_idx)
        self.stats = dict(ds_train.stats)
        
        self.label_key = label_key
        self.eval_model = eval_model
        assert eval_model in ['lin_reg', 'dec_tree_reg']
        self.metric = metric
        assert metric in ['rmse', 'rmspe']
        self.comparison = comparison
        assert comparison in ['ratio', 'margin']
        
        self.batch_size = batch_size
        self.device = device

        # Real data to be used in eval_model training.
        self.real_data = np.array([])
        dl = ds_train.get_dataloader(batch_size=batch_size, drop_last=False)
        for batch in dl:
            vec = regularize_vec(batch.numpy(), self.key_to_idx, self.stats)
            if self.real_data.shape[0] == 0:
                self.real_data = vec
            else:
                self.real_data = np.concatenate((self.real_data, vec), axis=0)
        self.feat_dim = self.real_data.shape[-1]

        # Portion of real data to be used for assessing two eval_models.
        self.test_data = np.array([])
        dl = ds_test.get_dataloader(batch_size=batch_size, drop_last=False)
        for batch in dl:
            vec = regularize_vec(batch.numpy(), self.key_to_idx, self.stats)
            if self.test_data.shape[0] == 0:
                self.test_data = vec
            else:
                self.test_data = np.concatenate((self.test_data, vec), axis=0)

    def evaluate(self, gen, latent_dim):
        gen.eval()
        fake_data = np.array([])
        while fake_data.shape[0] < self.n_train:
            b_size = min(self.n_train - fake_data.shape[0], self.batch_size)
            z = torch.randn(b_size, latent_dim, device=self.device)
            vec = gen(z).detach().numpy()[:, :self.feat_dim]
            vec = regularize_vec(vec, self.key_to_idx, self.stats)
            if fake_data.shape[0] == 0:
                fake_data = vec
            else:
                fake_data = np.concatenate((fake_data, vec), axis=0)
        
        idx = self.key_to_idx[self.label_key]
        real_label = self.real_data[:, idx]
        real_data = np.concatenate(
            (self.real_data[:, :idx], self.real_data[:, (idx+1):]), axis=1
        )
        fake_label = fake_data[:, idx]
        fake_data = np.concatenate(
            (fake_data[:, :idx], fake_data[:, (idx+1):]), axis=1
        )
        test_label = self.test_data[:, idx]
        test_data = np.concatenate(
            (self.test_data[:, :idx], self.test_data[:, (idx+1):]), axis=1
        )

        # For simple and quick evaluation, using simple regression models.
        if self.eval_model == 'lin_reg':
            real_model = LinearRegression().fit(real_data, real_label)
            fake_model = LinearRegression().fit(fake_data, fake_label)
        elif self.eval_model == 'dec_tree_reg':
            real_model = DecisionTreeRegressor().fit(real_data, real_label)
            fake_model = DecisionTreeRegressor().fit(fake_data, fake_label)
        
        real_pred = real_model.predict(test_data)
        fake_pred = fake_model.predict(test_data)
    
        if self.metric == 'rmse':
            real_metric = self.metric_rmse(real_pred, test_label)
            fake_metric = self.metric_rmse(fake_pred, test_label)
        elif self.metric == 'rmspe':
            real_metric = self.metric_rmspe(real_pred, test_label)
            fake_metric = self.metric_rmspe(fake_pred, test_label)
        
        if self.comparison == 'ratio':
            result = fake_metric / real_metric
        elif self.comparison == 'margin':
            result = fake_metric - real_metric
        
        return result, (fake_metric, real_metric)
    
    def metric_rmse(self, pred, label):
        return np.sqrt(((pred - label) * (pred - label)).mean())

    def metric_rmspe(self, pred, label):
        nonzero = (label > 0)
        err = (pred[nonzero] - label[nonzero]) / label[nonzero]
        return np.sqrt((err * err).mean())