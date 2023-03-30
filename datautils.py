import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')
    if univar:
        data = data[: -1:]

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)  # p_d 尝试解析index为日期格式
    dt_embed = _get_time_features(data.index)  # (1, time-series, variables)
    n_covariate_cols = dt_embed.shape[-1]

    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]  # target forecasting: oil temperature
        elif name == 'electricity':
            data = data[['MT_001']]
        elif name == 'WTH':
            data = data[['WetBulbCelsius']]
        else:
            data = data.iloc[:, -1:]

    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12 * 30 * 24)  # data[slice]: 去data的行
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    elif name.startswith('M5'):
        train_slice = slice(None, int(0.8 * (1913 + 28)))
        valid_slice = slice(int(0.8 * (1913 + 28)), 1913 + 28)
        test_slice = slice(1913 + 28 - 1, 1913 + 2 * 28)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])  # 计算矩阵每一列平均值和方差
    data = scaler.transform(data)  # 根据均值和方差，将矩阵转标准化
    if name in ('electricity') or name.startswith('M5'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)  # 1, ts, variables

    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed),
                                  0)  # data axis=-1的最后一维度是预测量，前7个是协变量
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)  # 把data、dt_embed按照最后一个维度concat

    if name in ('ETTh1', 'ETTh2', 'electricity', 'WTH'):
        pred_lens = [24, 48, 168, 336, 720]
    elif name.startswith('M5'):
        pred_lens = [28]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_forecast_bearing(name, univar=False, sc=True):  # sc for signal channel. T for horizontal (F -- vertical)
    n_covariate_cols = 0
    data = np.load('/data/yfy/FD-data/RUL/xjtu2_2560.npy', allow_pickle=True).item()
    # data = np.load('/data/yfy/FD-data/RUL/phm_dict.npy', allow_pickle=True).item()
    data = data['Bearing2_2'].astype(float)   # num, 2560, 2

    # data = np.load('/data/yfy/FD-data/RUL/phm_14.npy', allow_pickle=True).item()
    # data = data['Bearing1_4'].astype(float)

    # data = np.load('/data/yfy/FD-data/RUL/xjtu_35.npy', allow_pickle=True).item()
    # data = data['Bearing3_5'].astype(float)

    if univar:
        if sc:
            data = data[:, :, 0]  # horizontal channel
        else:
            data = data[:, :, 1]  # vertical accelerate channel

    data = data.reshape(-1, 2)

    scaler = StandardScaler().fit(data)  # 计算矩阵每一列平均值和方差
    data = scaler.transform(data)  # 根据均值和方差，将矩阵转标准化
    data = np.expand_dims(data, 0)

    pred_lens = [1, 2, 5, 8]

    return data, scaler, pred_lens, n_covariate_cols