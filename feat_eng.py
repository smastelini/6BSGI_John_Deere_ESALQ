import math
import pickle
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import cluster
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = (y_true - y_pred)
    mbe = diff.mean()
    return mbe


class MyPipeline:
    def __init__(self):
        self._scaler_clus = None
        self._kmeans = None
        self._scaler_model = None
        self._rf = None
        self._gbt_lower = None
        self._gbt_upper = None
        self._gaussian_process = None

        self._n_clusters = None
        self._unscaled_centers = None

    def _step0(self, data):
        return data.loc[:, ['POINT_X', 'POINT_Y']]

    def _step1(self, data):
        scaled_data = self._scaler_clus.transform(data)
        labels = self._kmeans.predict(scaled_data)
        dummies = pd.get_dummies(pd.Series(labels))

        return pd.concat([data, dummies], axis=1)

    def _step2(self, data):
        # Get view containing only the points
        data_pts = self._step0(data)
        unscaled_centers = self._unscaled_centers

        # Cluster dists
        # Euclidean
        for i in range(self._n_clusters):
            data[f'dist_e_{i}'] = np.sqrt(
                np.sum((data_pts.values - unscaled_centers[i]) ** 2, axis=1)
            )

        # Manhattan
        for i in range(self._n_clusters):
            data[f'dist_m_{i}'] = np.sum(
                np.abs(data_pts.values - unscaled_centers[i]), axis=1
            )

        return data

    @staticmethod
    def _dist_ratio(dists):
        return dists[0] / dists[1]

    @staticmethod
    def _get_angle_closest_center(points, center):
        dy = points[:, 1] - center[1]
        dx = points[:, 0] - center[0]
        return np.arctan2(dy, dx)

    def _step3(self, data):
        # cluster dist ratio
        tmp_view = data.loc[:, [f'dist_e_{i}' for i in range(self._n_clusters)]]
        data['dist_r'] = tmp_view.apply(lambda x: self._dist_ratio(sorted(x)), axis=1)

        data_pts = self._step0(data)
        # Angle between point and centers
        for i in range(self._n_clusters):
            data[f'angle_center_{i}'] = self._get_angle_closest_center(
                data_pts.values, self._unscaled_centers[i]
            )

        return data

    def _step4(self, data):
        return self._scaler_model.transform(data)

    def _step5_pred(self, data, pred_type):
        if pred_type == 'quantile':
            preds_lower = self._gbt_lower.predict(data)
            preds = self._rf.predict(data)
            preds_upper = self._gbt_upper.predict(data)

            return preds_lower, preds, preds_upper
        else:
            preds, preds_std = self._gaussian_process.predict(
                data, return_std=True
            )
            return preds, preds_std

    def run(self, data, pred_type='quantile'):
        import copy
        data = copy.deepcopy(data)

        data = self._step0(data)
        data = self._step1(data)
        data = self._step2(data)
        data = self._step3(data)
        data = self._step4(data)
        preds = self._step5_pred(data, pred_type)

        return preds


def make_feature_eng_pipeline(data_tr, data_ts, target, grupo, n_clusters):
    pipeline = MyPipeline()

    tr_y = data_tr[target]
    ts_y = data_ts[target]

    # Save view for later
    tr_x = pipeline._step0(data_tr)
    ts_x = pipeline._step0(data_ts)

    # Fit the kmeans model
    scaler_clus = preprocessing.StandardScaler()
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaler_clus.fit_transform(tr_x))

    # Save important data in the pipeline buffer
    pipeline._scaler_clus = scaler_clus
    pipeline._kmeans = kmeans
    pipeline._n_clusters = n_clusters
    pipeline._unscaled_centers = scaler_clus.inverse_transform(
        kmeans.cluster_centers_
    )

    # Assign dummies
    tr_x = pipeline._step1(tr_x)
    ts_x = pipeline._step1(ts_x)

    tr_x = pipeline._step2(tr_x)
    ts_x = pipeline._step2(ts_x)

    tr_x = pipeline._step3(tr_x)
    ts_x = pipeline._step3(ts_x)

    pd.concat([tr_x, tr_y], axis=1).to_csv(
        f'data/calibration_kmeans{n_clusters}_{grupo}_{target}.csv', index=False
    )
    pd.concat([ts_x, ts_y], axis=1).to_csv(
        f'data/validation_kmeans{n_clusters}_{grupo}_{target}.csv', index=False
    )

    scaler_model = preprocessing.StandardScaler()
    scaler_model.fit(tr_x)
    pipeline._scaler_model = scaler_model

    tr_x = pipeline._step4(tr_x)
    ts_x = pipeline._step4(ts_x)

    ########################
    #   Modelling portion  #
    ########################
    # RF + Quantile GBT
    gbt_lower = ensemble.GradientBoostingRegressor(
        n_estimators=100, random_state=42, subsample=0.7, loss="quantile", alpha=0.05
    )
    rf = ensemble.RandomForestRegressor(random_state=42)
    gbt_upper = ensemble.GradientBoostingRegressor(
        n_estimators=100, random_state=42, subsample=0.7, loss="quantile", alpha=0.95
    )

    gbt_lower.fit(tr_x, tr_y)
    rf.fit(tr_x, tr_y)
    gbt_upper.fit(tr_x, tr_y)

    pipeline._gbt_lower = gbt_lower
    pipeline._rf = rf
    pipeline._gbt_upper = gbt_upper

    preds_lower, preds, preds_upper = pipeline._step5_pred(ts_x, 'quantile')

    print('RF + Quantile GBT:')
    print('MAE = ', mean_absolute_error(ts_y, preds))
    print('RMSE = ', math.sqrt(mean_squared_error(ts_y, preds)))
    print('BIAS = ', MBE(ts_y, preds))
    print('COR = ', np.corrcoef(ts_y, preds)[0, 1])
    print('MAPE = ', mean_absolute_percentage_error(ts_y, preds) / 100.0)

    preds_inter = pd.DataFrame(
        np.zeros((len(data_ts), 4)), columns=['y_true', 'y_pred_05', 'y_pred', 'y_pred_95']
    )
    preds_inter['y_true'] = ts_y
    preds_inter['y_pred_05'] = preds_lower
    preds_inter['y_pred'] = preds
    preds_inter['y_pred_95'] = preds_upper

    # GPR
    kernel = Matern(5) + WhiteKernel()
    # kernel = None
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True
    )
    gpr.fit(tr_x, tr_y)

    pipeline._gaussian_process = gpr

    preds_gpr = pd.DataFrame(np.zeros((len(data_ts), 3)), columns=['y_true', 'y_pred', 'y_std'])

    preds, preds_std = pipeline._step5_pred(ts_x, 'gaussian_process')

    preds_gpr['y_true'] = ts_y
    preds_gpr['y_pred'] = preds
    preds_gpr['y_std'] = preds_std

    print('\nGPR:')
    print('MAE = ', mean_absolute_error(ts_y, preds))
    print('RMSE = ', math.sqrt(mean_squared_error(ts_y, preds)))
    print('BIAS = ', MBE(ts_y, preds))
    print('COR = ', np.corrcoef(ts_y, preds)[0, 1])
    print('MAPE = ', mean_absolute_percentage_error(ts_y, preds) / 100.0)

    return preds_inter, preds_gpr, pipeline


if __name__ == '__main__':
    print('\ngrupo A')
    data_tr = pd.read_csv('data/calibration_data_group_a.csv')
    data_ts = pd.read_csv('data/validation_data_group_a.csv')

    rf_gbt, gpr, pipeline_A = make_feature_eng_pipeline(data_tr, data_ts, 'Ca', 'A', 10)

    rf_gbt.to_csv(
        'predictions/RF+GBT_preds_and_intervals_Ca_A.csv', index=False
    )
    gpr.to_csv('predictions/GPR_preds_and_std_Ca_A.csv', index=False)

    print('\ngrupo B')
    data_tr = pd.read_csv('data/calibration_data_group_b.csv')
    data_ts = pd.read_csv('data/validation_data_group_b.csv')

    rf_gbt, gpr, pipeline_B = make_feature_eng_pipeline(data_tr, data_ts, 'Ca', 'B', 10)

    rf_gbt.to_csv(
        'predictions/RF+GBT_preds_and_intervals_Ca_B.csv', index=False
    )
    gpr.to_csv('predictions/GPR_preds_and_std_Ca_B.csv', index=False)

    # Save models
    with open('models/pipeline_A.mdl', 'wb') as f:
        pickle.dump(pipeline_A, f, pickle.HIGHEST_PROTOCOL)

    with open('models/pipeline_B.mdl', 'wb') as f:
        pickle.dump(pipeline_B, f, pickle.HIGHEST_PROTOCOL)
