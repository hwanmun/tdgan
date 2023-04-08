import os
import pandas as pd
import numpy as np
import datetime
import torch

def split_sales_dataset(store_file, sales_file, path, name,
                        split=[0.8, 0.1, 0.1], random_seed=None):
    """ Combine store.csv and train.csv.
    Split the combined data into train, dev, test datasets.
    """
    assert np.isclose(sum(split), 1), 'Split portions should sum up to 1.'

    store = pd.read_csv(store_file, index_col='Store')
    store_cols = store.columns.values.tolist()
    store = store.to_dict('index')
    
    sales = pd.read_csv(sales_file, dtype={'StateHoliday': str})
    for col in store_cols:
        sales[col] = sales['Store'].map({k: v[col] for k, v in store.items()})

    seed = random_seed
    train = sales.sample(frac=split[0], random_state=seed)
    sales = sales.drop(train.index)
    dev, test = [], []
    if len(sales) > 0:
        if seed is not None:
            seed += 1
        dev = sales.sample(frac=split[1]/sum(split[1:]), random_state=seed)
        test = sales.drop(dev.index)
    
    if not os.path.exists(path):
        os.mkdir(path)
    for data, split in zip([train, dev, test], ['train', 'dev', 'test']):
        if len(data) > 0:
            file_path = os.path.join(path, f'{name}_{split}.csv')
            data.to_csv(file_path, index=False)
            print(f'{split} dataset created at {file_path}, '
                  f'contains {len(data)} entries.')
   

class SalesDataset(torch.utils.data.Dataset):
    """dataset class to process tabular data and turn them into tensors."""
    def __init__(self, file_path, num_stores=1115, store_fmt='onehot',
                 dayofweek_fmt='periodic', date_fmt='periodic', stats=None,
                 stats_from=None, sales_only=False, noise_size=0.02
                 ):
        self.data = pd.read_csv(file_path)
        self.num_stores = num_stores
        self.store_fmt = store_fmt
        self.dayofweek_fmt = dayofweek_fmt
        self.date_fmt = date_fmt
        assert self.store_fmt in ['onehot', 'numeral']
        assert self.dayofweek_fmt in ['periodic', 'onehot']
        assert self.date_fmt in ['periodic', 'numeral']
        self.sales_only = sales_only
        self.noise_size = noise_size

        # Store field (1~1115) into one-hot vector or numeral (-1~1) value.
        if self.store_fmt == 'numeral':
            self.data['StoreNum'] = 2 * ((self.data['Store'] - 1)
                                         / (self.num_stores - 1)) - 1

        # DayOfWeek (1~7) into one-hot vector or peridoic cos/sin values.
        if self.dayofweek_fmt == 'periodic':
            dow = self.data['DayOfWeek'] / 7
            self.data['DayOfWeekX'] = np.cos(2 * np.pi * dow)
            self.data['DayOfWeekY'] = np.sin(2 * np.pi * dow)

        dt = pd.to_datetime(self.data['Date'])
        yr = dt.apply(lambda d: d.year)
        wk = dt.apply(lambda d: d.week)
        mo = dt.apply(lambda d: d.month)
        day = dt.apply(
            lambda d: (d.timetuple().tm_yday
                       / datetime.date(d.year, 12, 31).timetuple().tm_yday)
        )

        # Date into floating numeral value or integer yr + perioic day values.
        if self.date_fmt == 'periodic':
            self.data['DateYr'] = yr
            self.data['DateX'] = np.cos(2 * np.pi * day)
            self.data['DateY'] = np.sin(2 * np.pi * day)
        elif self.date_fmt == 'numeral':
            self.data['DateNum'] = yr + day

        # StateHoliday (0,a,b,c) into 3-dim one-hot vector.
        holiday = self.data['StateHoliday']
        self.data['StateHolidayPublic'] = (holiday == 'a').astype(int)
        self.data['StateHolidayEaster'] = (holiday == 'b').astype(int)
        self.data['StateHolidayChristmas'] = (holiday == 'c').astype(int)
        
        # Assortment (a~c) into integer, -1, 0, 1.
        ast = self.data['Assortment']
        self.data['Assortment'] = ((ast=='c') * 1 - (ast=='a') * 1).astype(int)

        # CompetitionDistance NaN is replaced with maximum distance reported.
        max_val = self.data['CompetitionDistance'].max()
        replaced = self.data['CompetitionDistance'].replace(np.nan, max_val)
        self.data['CompetitionDistance'] = replaced

        # CompetitionOpenSince into total months since the competition opening.
        compete = (mo - self.data['CompetitionOpenSinceMonth']
                   + 12 * (yr - self.data['CompetitionOpenSinceYear']))
        self.data['Competition'] = (compete > 0).astype(int)
        self.data['CompetitionMonths'] = (compete * (compete > 0)).fillna(0)

        # Promo2Since into total weeks since the promo2 sign-up.
        promo2 = (wk - self.data['Promo2SinceWeek']
                  + 52 * (yr - self.data['Promo2SinceYear']))
        self.data['Promo2Weeks'] = (promo2 * self.data['Promo2']).fillna(0)
        
        # PromoInterval into 3-dim one-hot vector.
        pintv = self.data['PromoInterval']
        self.data['Promo2Jan'] = (pintv == 'Jan,Apr,Jul,Oct').astype(int)
        self.data['Promo2Feb'] = (pintv == 'Feb,May,Aug,Nov').astype(int)
        self.data['Promo2Mar'] = (pintv == 'Mar,Jun,Sept,Dec').astype(int)

        self.data = self.data.drop(
            columns=['Promo2SinceWeek', 'Promo2SinceYear',
                     'PromoInterval', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear']
        )

        # location of each field in a vector is specified in key_to_idx.
        keys = []
        if self.store_fmt == 'onehot':
            keys += [f'Store_{i+1}' for i in range(self.num_stores)]
        elif self.store_fmt == 'numeral':
            keys += ['StoreNum']
        if self.dayofweek_fmt == 'onehot':
            keys += [f'DayOfWeek_{i+1}' for i in range(7)]
        elif self.dayofweek_fmt == 'periodic':
            keys += ['DayOfWeekX', 'DayOfWeekY']
        if self.date_fmt == 'periodic':
            keys += ['DateYr', 'DateX', 'DateY']
        elif self.date_fmt == 'numeral':
            keys += ['DateNum']
        keys += ['Sales', 'Customers', 'Open', 'Promo', 'StateHolidayPublic',
                 'StateHolidayEaster', 'StateHolidayChristmas', 'SchoolHoliday']
        if not self.sales_only:
            keys += ['Assortment', 'CompetitionDistance', 'Promo2']
            keys += [f'StoreType_{v}' for v in ['a', 'b', 'c', 'd']]
            keys += ['Competition', 'CompetitionMonths', 'Promo2Weeks',
                     'Promo2Jan', 'Promo2Feb', 'Promo2Mar']
        self.key_to_idx = {key: i for i, key in enumerate(keys)}
        self.feat_dim = len(keys)
        
        # Calculate mean/std of unbounded numerical values to normalize them.
        self.stats = {}
        if stats is not None:
            self.stats = stats
        elif stats_from is not None:
            self.copy_stats_from(stats_from)
        else:
            self.calculate_stats()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.loc[idx, :].to_dict()
    
    def calculate_stats(self):
        keys = ['Sales', 'Customers', 'CompetitionDistance',
                'CompetitionMonths', 'Promo2Weeks']
        if self.date_fmt == 'periodic':
            keys.append('DateYr')
        elif self.date_fmt == 'numeral':
            keys.append('DateNum')

        for key in keys:
            self.stats[key] = (self.data[key].mean(), self.data[key].std())
    
    def copy_stats_from(self, target):
        assert self.date_fmt == target.date_fmt, 'Format mismatch.'
        assert self.dayofweek_fmt == target.dayofweek_fmt, 'Format mismatch.'
        self.stats = dict(target.stats)

    def collate_vec(self, batch):
        vec = torch.zeros(len(batch), self.feat_dim)
        for row, sample in enumerate(batch):
            for key, val in sample.items():
                if key in ['Store', 'DayOfWeek', 'StoreType']:
                    key = f'{key}_{val}'
                    if key in self.key_to_idx:
                        vec[row, self.key_to_idx[key]] = 1
                else:
                    if key in self.stats:
                        mean, std = self.stats[key]
                        val = (val - mean) / std
                    if key in self.key_to_idx:
                        vec[row, self.key_to_idx[key]] = val
        return vec
    
    # Add noise to outputing vector to make generative model training eaiser.
    def collate_noise(self, batch):
        noise = torch.randn(len(batch), self.feat_dim) * self.noise_size
        return self.collate_vec(batch) + noise
    
    def get_dataloader(self, **kwargs):
        if 'collate_fn' not in kwargs or kwargs['collate_fn'] == 'vec':
            kwargs['collate_fn'] = self.collate_vec
        elif kwargs['collate_fn'] == 'noise':
            kwargs['collate_fn'] = self.collate_noise
        return torch.utils.data.DataLoader(self, **kwargs)