import numpy as np
import pandas as pd

def vec_to_table(org_vec, key_to_idx, ds_stats):
    """ Transform (unregularized) vectors into pandas DataFrame. """
    vec = regularize_vec(org_vec, key_to_idx, ds_stats)
    n_store = len([k for k in key_to_idx.keys() if k.split('_')[0] == 'Store'])
    data = {}
    
    if 'StoreNum' in key_to_idx:
        idx = key_to_idx['StoreNum']
        data['Store'] = vec[:, idx].astype(int)
    else:
        data['Store'] = np.argmax(vec[:, :n_store], axis=1) + 1
    
    if 'DayOfWeekX' in key_to_idx:
        i_x, i_y = key_to_idx['DayOfWeekX'], key_to_idx['DayOfWeekY']
        day = np.angle(vec[:, i_x] + 1j * vec[:, i_y]) * (7 / (2 * np.pi))
        day = np.mod(np.round(day), 7).astype(int)
        data['DayOfWeek'] = (day + (day == 0) * 7).astype(int)
    else:
        idxs = [v for k, v in key_to_idx.items()
                if k.split('_')[0] == 'DayofWeek']
        i1, i2 = min(idxs), max(idxs) + 1
        data['DayOfWeek'] = (np.argmax(vec[:, i1:i2], axis=1) + 1).astype(int)
    
    if 'DateNum' in key_to_idx:
        idx = key_to_idx['DateNum']
        yr, day = np.floor(vec[:, idx]), np.mod(vec[:, idx], 1.0)
    else:
        idx = key_to_idx['DateYr']
        yr = vec[:, idx]
        i_x, i_y = key_to_idx['DateX'], key_to_idx['DateY']
        ang = np.angle(vec[:, i_x] + 1j * vec[:, i_y]) / np.pi
        day = np.mod(ang, 2.0) / 2
    ydays = ((yr.astype(int) - 1969).astype('datetime64[Y]')
             + np.timedelta64(0, 'D')
             - (yr.astype(int) - 1970).astype('datetime64[Y]'))
    day = np.round(day * ydays.astype(int))
    date = ((yr.astype(int) - 1970).astype('datetime64[Y]')
            + (day-1).astype('timedelta64[D]'))
    data['Date'] = date.astype(str)

    for key in ['Sales', 'Customers', 'Open', 'Promo']:
        idx = key_to_idx[key]
        data[key] = vec[:, idx].astype(int)
    i1 = key_to_idx['StateHolidayPublic']
    i2 = key_to_idx['StateHolidayChristmas'] + 1
    st_hd = vec[:, i1:i2]
    st_hd = (np.argmax(st_hd, axis=1) + 1) * (st_hd.sum(axis=1) > 0)
    st_hd = st_hd.astype(int).astype(str)
    for v1, v2 in [('1', 'a'), ('2', 'b'), ('3', 'c')]:
        st_hd[st_hd == v1] = v2
    data['StateHoliday'] = st_hd
    data['SchoolHoliday'] = vec[:, key_to_idx['SchoolHoliday']].astype(int)

    df = pd.DataFrame(data=data)
    return df


def regularize_vec(org_vec, key_to_idx, ds_stats):
    """ Restore unregularized vector back to regularized vector.
    Ex: restore integer categorical values, restore numerical values from
    normalized values using stats stored in the dataset class instance.
    """
    vec = np.copy(org_vec)
    n_store = len([k for k in key_to_idx.keys() if k.split('_')[0] == 'Store'])
    if 'StoreNum' in key_to_idx:
        idx = key_to_idx['StoreNum']
        vec[:, idx] = np.round((vec[:, idx] + 1) * (n_store - 1) / 2) + 1
    if 'DayOfWeekX' in key_to_idx:
        i_x, i_y = key_to_idx['DayOfWeekX'], key_to_idx['DayOfWeekY']
        ang = np.angle(vec[:, i_x] + 1j * vec[:, i_y])
        ang = np.round(ang * (7 / (2 * np.pi))) * (2 * np.pi / 7)
        vec[:, i_x], vec[:, i_y] = np.cos(ang), np.sin(ang)
    if 'DateX' in key_to_idx:
        i_x, i_y = key_to_idx['DateX'], key_to_idx['DateY']
        r = np.sqrt(vec[:, i_x] * vec[:, i_x] + vec[:, i_y] * vec[:, i_y])
        vec[:, i_x] = vec[:, i_x] / r
        vec[:, i_y] = vec[:, i_y] / r
    if 'Assortment' in key_to_idx:
        idx = key_to_idx['Assortment']
        vec[:, idx] = np.round(vec[:, idx])
    for key, (mean, std) in ds_stats.items():
        if key not in key_to_idx:
            continue
        idx = key_to_idx[key]
        if key == 'DateNum':
            vec[:, idx] = vec[:, idx] * std + mean
        else:    
            vec[:, idx] = np.round(vec[:, idx] * std + mean)
    
    for key, idx in key_to_idx.items():
        if key in ['StoreNum', 'DayOfWeekX', 'DayOfWeekY', 'DateX', 'DateY']:
            continue
        if key in ds_stats:
            continue
        vec[:, idx] = (vec[:, idx] > 0.5).astype(int)
    
    return vec
