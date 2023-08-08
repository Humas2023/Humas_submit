import os
import time

import torch

from sklearn.mixture import GaussianMixture as GMM
from alibi_detect.cd import MMDDrift, LSDDDrift, ContextMMDDrift
from alibi_detect.utils.pytorch import GaussianRBF
import json
from alibi_detect.saving import save_detector, load_detector
#

from xgboost import XGBRegressor as XGBR
from heter_predict_model import UsageMapPredictor

from utils import MAPE, preprocess_qps_usage_data
from sklearn.ensemble import RandomForestRegressor as RFR  # 随机森林模块
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR  # 线性回归模块

import pandas as pd
import numpy as np


# 自动聚类类别数确定算法：
#     对提取的feature进行聚类，聚类类别的上限根据BIC算法自动确定：对于从分为2类开始，设定不同类别上限来训练GMM模型，
# 当训练后模型的BIC指标的相对变化连续较小（默认为连续小于0.15）时，即认为寻找到合适的分类类别，
# 选用BIC指标是因为相对于AIC，BIC更容易选择简单的模型，即倾向于选择较少的类别数，
# 因为起始时刻条目有限，所以无需选择较大数量的分类类别数
def find_n_components_gmm(bic_aic_pdf, select_n='bic', diff_limit=0.15, diff_win=3, com_col='component'):
    '''
        bic_aic_pdf: 不同类别数对应的模型AIC，BIC指标的dataframe
        select_n: 选择哪种指标来进行n_components的选取，默认为bic
        diff_limit: 随着n_components上升导致的BIC（或AIC）指标的相对变化的绝对值的上限，若小于该上限则认为模型变化很小，不需要增加聚类的类别数量
        diff_win: 若当前n_components连续连续递增diff_win次（即对于[n_components+1，n_components+diff_win]）也没有出现相对变化超过diff_limit的情况，则认为当前n_components已经满足要求
    '''
    bic_aic = bic_aic_pdf.copy().reset_index(drop=True)

    bic_aic['%s_diff' % select_n] = (bic_aic[select_n].diff() / bic_aic[select_n]).abs()
    bic_aic = bic_aic.dropna().sort_values(com_col).reset_index(drop=True)
    n_components_res = -1

    for i in range(bic_aic.shape[0]):
        if n_components_res > 0:
            break
        if i < bic_aic.shape[0] - diff_win:
            tmp_max_diff = (bic_aic['%s_diff' % select_n][i + 1:i + diff_win + 1]).max()
            if tmp_max_diff <= diff_limit:
                n_components_res = bic_aic[com_col][i]
                break
        elif i < bic_aic.shape[0]:
            tmp_max_diff = (bic_aic['%s_diff' % select_n][i + 1:]).max()
            if tmp_max_diff <= diff_limit:
                n_components_res = bic_aic[com_col][i]
                break

    if n_components_res < 0:
        n_components_res = np.max([bic_aic.component.quantile(0.5), 2])

    if np.isnan(np.array(n_components_res, dtype=np.float32)):
        print(bic_aic)
        print(bic_aic_pdf)

    return n_components_res, bic_aic


def load_config(config_file):
    f = open(config_file, encoding='utf-8')
    res = f.read()
    config_content = json.loads(res)
    f.close()
    return config_content


def save_config(config, filename):
    config_content = {}
    for key, value in config.items():
        config_content[key] = value

    fw = open(filename, 'w', encoding='utf-8')

    dic_json = json.dumps(config_content, ensure_ascii=False, indent=4)
    fw.write(dic_json)
    fw.close()


def find_drift_results(x_df_dict, base_model="8269CY", unit="ds",
                       x_col="qps__Q_s", y_col='cpu_usage', freq='t', ds="20220815",
                       base_dir="./results_lsdd_no_update", app="MS0", pdc_col='RUE',
                       sample_time="sample_time_n",
                       x_unit=1, y_unit=60, block_size=1, cov_col="cpu_util__pct", threshold_col="rue"):
    total_file_base = "total_cor_%s" % base_model

    pio_ten_files = os.listdir(os.path.join(os.path.join(base_dir, app), ds))

    aim_files = []
    for ptf in pio_ten_files:
        if total_file_base in ptf and ".csv" in ptf:
            aim_files.append(ptf)

    total_results = []

    for ptf in aim_files:
        tmp_res = pd.read_csv(os.path.join(os.path.join(os.path.join(base_dir, app), ds), ptf))
        # start,end,qps_usage_corr,step,label,p-value,p-value-thres,distance,distance-thres
        tmp_res.start = pd.to_datetime(tmp_res.start)
        tmp_res.end = pd.to_datetime(tmp_res.end)
        tmp_res.label = tmp_res.label.astype(int)

        tmp_res = tmp_res.sort_values("start", ascending=False).reset_index(drop=True)
        total_results.append(tmp_res)

    aim_res = total_results[0]

    for i in range(len(total_results)):
        if i < 1:
            continue
        aim_total_pdf = total_results[i][total_results[i].label == 0]
        aim_last_time = aim_total_pdf.start.max()
        now_aim_last_time = aim_res[aim_res.label == 0].start.max()

        if now_aim_last_time < aim_last_time:
            aim_res = total_results[i].reset_index(drop=True)

    aim_no_drift_pdf = aim_res[aim_res.label == 0].sort_values("start", ascending=False).reset_index(drop=True)

    aim_drift_pdf = aim_res[aim_res.label == 1].sort_values("start", ascending=False).reset_index(drop=True)

    base_pdf = aim_no_drift_pdf.head(1).reset_index(drop=True)

    aim_no_drift_shape = aim_no_drift_pdf.shape[0]

    aim_no_drift_pdf = aim_no_drift_pdf.tail(aim_no_drift_shape - 1).sort_values("start", ascending=False).reset_index(
        drop=True)

    total_corr = {"start": [], "end": [], threshold_col: []}

    in_df = x_df_dict[base_model].copy()
    in_df = in_df.sort_values(sample_time).reset_index(drop=True)
    in_df[x_col] = in_df[x_col] * x_unit
    in_df[y_col] = in_df[y_col] * y_unit

    aim_res_total_pdf = pd.concat([base_pdf, aim_no_drift_pdf, aim_drift_pdf], axis=0).sort_values("start",
                                                                                                   ascending=False).reset_index(
        drop=True)

    unit_range = in_df[unit].unique().tolist()

    unit_range.sort(reverse=False)

    for i in range(aim_res_total_pdf.shape[0]):
        total_corr["start"].append(base_pdf.start.values[i])
        total_corr["end"].append(base_pdf.end.values[i])
        # total_corr = {'start': [], "end": [], 'qps_usage_corr': [], 'step': []}

        now_df_list = {}

        for m in x_df_dict.keys():
            if m == base_model:
                now_df = in_df[(in_df[sample_time] >= base_pdf.start.values[i])
                               & (in_df[sample_time] <= base_pdf.end.values[i])].reset_index(drop=True)

                now_df_list[m] = now_df
                now_df_list[m]["map"] = 1.0
                # now_df_list[m]["pred_map"] = 1.0
                now_df_list[m]["base_%s" % y_col] = now_df_list[m][y_col].copy()
                now_df_list[m]["aim_%s" % y_col] = now_df_list[m][y_col].copy()
                now_df_list[m]["base_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                now_df_list[m]["aim_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                now_df_list[m]["base_%s" % cov_col] = now_df_list[m][cov_col].copy()
                now_df_list[m]["base_%s" % x_col] = now_df_list[m][x_col].copy()
            else:
                now_base_df = in_df[(in_df[sample_time] >= base_pdf.start.values[i])
                                    & (in_df[sample_time] <= base_pdf.end.values[i])].reset_index(drop=True)
                now_raw_df = x_df_dict[m][(x_df_dict[m][sample_time] >= base_pdf.start.values[i])
                                          & (x_df_dict[m][sample_time] <= base_pdf.end.values[i])].reset_index(
                    drop=True)

                # now_base_df.rename(columns={cov_col:"base_%s" % cov_col},inplace=True)
                if now_raw_df.shape[0] < 1:
                    continue
                now_raw_df[x_col] = now_raw_df[x_col] * x_unit
                now_raw_df[y_col] = now_raw_df[y_col] * y_unit

                now_base_df["base_%s" % y_col] = now_base_df[y_col].copy()
                now_base_df["base_%s" % cov_col] = now_base_df[cov_col].copy()
                now_base_df["base_%s" % pdc_col] = now_base_df[pdc_col].copy()
                now_base_df["base_%s" % x_col] = now_base_df[x_col].copy()
                now_raw_df = pd.merge(now_base_df[
                                          [sample_time, "base_%s" % cov_col, "base_%s" % y_col, "base_%s" % pdc_col,
                                           "base_%s" % x_col]], now_raw_df, on=sample_time)
                if now_raw_df.shape[0] < 1:
                    continue
                now_raw_df = now_raw_df.sort_values(sample_time).reset_index(drop=True)
                now_base_df = now_base_df.sort_values(sample_time).reset_index(drop=True)

                now_raw_df["map"] = now_raw_df["base_%s" % pdc_col] / now_raw_df[pdc_col]

                # maps = perf_map.predict_ll(now_raw_df[["base_%s" % cov_col]].to_numpy())
                # now_raw_df["pred_map"] = pd.Series(maps)
                now_raw_df["aim_%s" % y_col] = now_raw_df[y_col] * now_raw_df["map"]

                now_df_list[m] = now_raw_df
                print(MAPE(now_raw_df["base_%s" % y_col],
                           now_raw_df["aim_%s" % y_col] / now_raw_df[x_col] * now_raw_df["base_%s" % x_col]))

        aim_now_df = []
        for m in now_df_list.keys():
            if m == base_model:
                tmp_pdf = now_df_list[m][[sample_time, unit, x_col, y_col]].reset_index(drop=True)
                aim_now_df.append(tmp_pdf)
            else:
                tmp_pdf = now_df_list[m][[sample_time, unit, x_col, "aim_%s" % y_col]].reset_index(drop=True)
                tmp_pdf.rename(columns={"aim_%s" % y_col: y_col}, inplace=True)
                aim_now_df.append(tmp_pdf)

        now_df = pd.concat(aim_now_df, axis=0).reset_index(drop=True)
        now_df = now_df.dropna().reset_index(drop=True)

        total_corr[threshold_col].append((now_df[y_col] / now_df[x_col]).mean())

    total_corr_pdf = pd.DataFrame(total_corr)

    aim_res_total_pdf = pd.merge(aim_res_total_pdf, total_corr_pdf, on=["start", "end"]).sort_values("start",
                                                                                                     ascending=False).reset_index(
        drop=True)


def find_train_range_dist(x_df_dict, base_model="826X", init_size=1, freq='t', step=1,
                          diff_win=3, unit="ds",
                          model="context",
                          p_val=0.05,
                          if_update_ref=False,
                          x_ref_preprocessed=False,
                          preprocess_at_init=True,
                          inverse=True, n_permutations=100, lambda_rd_max=0.2, update_x_ref="reservoir_sampling",
                          update_ref="last",
                          backend="pytorch", prop_c_held=0.25, n_folds=5,
                          device="cpu", early_stop=True,
                          context_col=["minute"],
                          heter_context=True,
                          replicas_col="replicas",
                          heter_method="h",
                          sample_time='sample_time__m', norm=False, x_col="qps__Q_s", y_col='cpu_usage',
                          x_unit=1, y_unit=60, block_size=1,
                          standard=False, lateset_range=2, mode=1, norm_mode=1, cov_col="cpu_util__pct",
                          pdc_col='RUE', save=True, save_path="./results_lsdd_no_update/", use_gpu=True):
    '''
           backend: TensorFlow, PyTorch and KeOps implementations of the MMD detector are available. Specify the backend (tensorflow, pytorch or keops). Defaults to tensorflow.

           p_val: p-value used for significance of the permutation test.
           preprocess_at_init: Whether to already apply the (optional) preprocessing step to the reference data at initialization and store the preprocessed data. Dependent on the preprocessing step, this can reduce the computation time for the predict step significantly, especially when the reference dataset is large. Defaults to True. It is possible that it needs to be set to False if the preprocessing step requires statistics from both the reference and test data, such as the mean or standard deviation.
           x_ref_preprocessed: Whether or not the reference data x_ref has already been preprocessed. If True, the reference data will be skipped and preprocessing will only be applied to the test data passed to predict.
           update_x_ref: Reference data can optionally be updated to the last N instances seen by the detector or via reservoir sampling with size N. For the former, the parameter equals {‘last’: N} while for reservoir sampling {‘reservoir_sampling’: N} is passed.
           preprocess_fn: Function to preprocess the data before computing the data drift metrics. Typically a dimensionality reduction technique.
           kernel: Kernel used when computing the MMD. Defaults to a Gaussian RBF kernel (from alibi_detect.utils.pytorch import GaussianRBF, from alibi_detect.utils.tensorflow import GaussianRBF or from alibi_detect.utils.keops import GaussianRBF dependent on the backend used). Note that for the KeOps backend, the diagonal entries of the kernel matrices kernel(x_ref, x_ref) and kernel(x_test, x_test) should be equal to 1. This is compliant with the default Gaussian RBF kernel.
           sigma: Optional bandwidth for the kernel as a np.ndarray. We can also average over a number of different bandwidths, e.g. np.array([.5, 1., 1.5]).
           configure_kernel_from_x_ref: If sigma is not specified, the detector can infer it via a heuristic and set sigma to the median (TensorFlow and PyTorch) or the mean pairwise distance between 2 samples (KeOps) by default. If configure_kernel_from_x_ref is True, we can already set sigma at initialization of the detector by inferring it from x_ref, speeding up the prediction step. If set to False, sigma is computed separately for each test batch at prediction time.
           n_permutations: Number of permutations used in the permutation test.
           input_shape: Optionally pass the shape of the input data.
           data_type: can specify data type added to the metadata. E.g. ‘tabular’ or ‘image’.

           lambda_rd_max: The maximum relative difference between two estimates of LSDD that the regularization parameter lambda is allowed to cause. Defaults to 0.2 as in the paper.

           update_ref: Reference data can optionally be updated to the last N instances seen by the detector. The parameter should be passed as a dictionary {‘last’: N}.

           x_kernel: Kernel defined on the data x_*. Defaults to a Gaussian RBF kernel (from alibi_detect.utils.pytorch import GaussianRBF or from alibi_detect.utils.tensorflow import GaussianRBF dependent on the backend used).

           c_kernel: Kernel defined on the context c_*. Defaults to a Gaussian RBF kernel (from alibi_detect.utils.pytorch import GaussianRBF or from alibi_detect.utils.tensorflow import GaussianRBF dependent on the backend used).

           prop_c_held: Proportion of contexts held out to condition on.

           n_folds: Number of cross-validation folds used when tuning the regularisation parameters.

           batch_size: If not None, then compute batches of MMDs at a time rather than all at once which could lead to memory issues.

           input_shape: Optionally pass the shape of the input data.

           data_type: can specify data type added to the metadata. E.g. ‘tabular’ or ‘image’.

           verbose: Whether or not to print progress during configuration.

           Additional PyTorch keyword arguments:
           device: cuda or gpu to use the GPU and cpu for the CPU. If the device is not specified, the detector will try to leverage the GPU if possible and otherwise fall back on CPU.
    '''

    aim_model_x = {base_model: 0, "816X": 1}

    # save_config(aim_model_x,"model_label.json")
    if os.path.exists("model_label.json"):
        aim_model_x = load_config("model_label.json")
    else:
        save_config(aim_model_x, "model_label.json")

    if freq not in ['t', 'h', 's', 'd']:
        print("the freq is invalid: the valid set:['t','h','s','d'];")
        print("t: minute;")
        print("h: hour;")
        print("s: second;")
        print("d: day")

    if standard:
        total_corr_pdf_list = {}
        aim_corr_pdf_list = {}
        aim_times = {}
        bic_aic_results_list = {}
        # usage_data_list = {}

        # perf_alpha_list = {}

        in_df = x_df_dict[base_model].copy()
        in_df = in_df.sort_values(sample_time).reset_index(drop=True)
        in_df[x_col] = in_df[x_col] * x_unit
        in_df[y_col] = in_df[y_col] * y_unit
        total_corr = {'start': [], "end": [], 'qps_usage_corr': [], 'step': []}

        unit_range = in_df[unit].unique().tolist()

        unit_range.sort(reverse=inverse)

        df_list_data = {}

        base_mean_std = {"x_mean": in_df[x_col].mean(), "x_std": in_df[x_col].std(),
                         "y_mean": in_df[y_col].mean(), "y_std": in_df[y_col].std(),
                         "x_max": in_df[x_col].max(), "x_min": in_df[x_col].min(),
                         "y_max": in_df[y_col].max(), "y_min": in_df[y_col].min()
                         }

        detected_drifts = 0
        for i in range(0, len(unit_range), step):
            now_df_list = {}
            for m in x_df_dict.keys():
                if m == base_model:
                    if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                        now_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                        print(i)
                    else:
                        if i == 0:

                            now_df = in_df[(in_df[unit] <= unit_range[i])
                                           & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                        else:
                            now_df = in_df[(in_df[unit] <= unit_range[i])
                                           & (in_df[unit] > unit_range[i + block_size])].reset_index(drop=True)

                    now_df_list[m] = now_df
                    now_df_list[m]["map"] = 1.0
                    now_df_list[m]["pred_map"] = 1.0
                    now_df_list[m]["base_%s" % y_col] = now_df_list[m][y_col].copy()
                    now_df_list[m]["aim_%s" % y_col] = now_df_list[m][y_col].copy()
                    now_df_list[m]["base_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                    now_df_list[m]["aim_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                    now_df_list[m]["base_%s" % cov_col] = now_df_list[m][cov_col].copy()
                    now_df_list[m]["base_%s" % x_col] = now_df_list[m][x_col].copy()
                else:
                    if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                        now_base_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                        now_raw_df = x_df_dict[m][x_df_dict[m][unit] <= unit_range[i]].copy().reset_index(drop=True)
                        print(i)
                    else:
                        if i == 0:
                            now_base_df = in_df[(in_df[unit] <= unit_range[i])
                                                & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                            now_raw_df = x_df_dict[m][(x_df_dict[m][unit] <= unit_range[i])
                                                      & (x_df_dict[m][unit] > unit_range[i + init_size])].reset_index(
                                drop=True)
                        else:
                            now_base_df = in_df[(in_df[unit] <= unit_range[i])
                                                & (in_df[unit] > unit_range[i + block_size])].copy().reset_index(
                                drop=True)
                            now_raw_df = x_df_dict[m][(x_df_dict[m][unit] <= unit_range[i])
                                                      & (x_df_dict[m][unit] > unit_range[i + block_size])].reset_index(
                                drop=True)

                    # now_base_df.rename(columns={cov_col:"base_%s" % cov_col},inplace=True)
                    if now_raw_df.shape[0] < 1:
                        continue
                    now_raw_df[x_col] = now_raw_df[x_col] * x_unit
                    now_raw_df[y_col] = now_raw_df[y_col] * y_unit

                    now_base_df["base_%s" % y_col] = now_base_df[y_col].copy()
                    now_base_df["base_%s" % cov_col] = now_base_df[cov_col].copy()
                    now_base_df["base_%s" % pdc_col] = now_base_df[pdc_col].copy()
                    now_base_df["base_%s" % x_col] = now_base_df[x_col].copy()
                    now_raw_df = pd.merge(now_base_df[
                                              [sample_time, "base_%s" % cov_col, "base_%s" % y_col, "base_%s" % pdc_col,
                                               "base_%s" % x_col]], now_raw_df, on=sample_time)
                    if now_raw_df.shape[0] < 1:
                        continue
                    now_raw_df = now_raw_df.sort_values(sample_time).reset_index(drop=True)
                    now_base_df = now_base_df.sort_values(sample_time).reset_index(drop=True)

                    now_raw_df["map"] = now_raw_df["base_%s" % pdc_col] / now_raw_df[pdc_col]

                    perf_map = UsageMapPredictor()
                    perf_map.train(X_train=now_raw_df[["base_%s" % cov_col]].to_numpy(),
                                   y_train=now_raw_df["map"].to_numpy())

                    maps = perf_map.predict_ll(now_raw_df[["base_%s" % cov_col]].to_numpy())
                    now_raw_df["pred_map"] = pd.Series(maps)
                    now_raw_df["aim_%s" % y_col] = now_raw_df[y_col] * now_raw_df["pred_map"]

                    now_df_list[m] = now_raw_df
                    # print(now_raw_df.head(10))
                    print(MAPE(now_raw_df["base_%s" % y_col],
                               now_raw_df["aim_%s" % y_col] / now_raw_df[x_col] * now_raw_df["base_%s" % x_col]))
                    print(MAPE(now_raw_df["map"], now_raw_df["pred_map"]))

            if "h" in heter_method or base_model not in now_df_list.keys():
                aim_now_df = []

                for m in now_df_list.keys():
                    if m == base_model:
                        tmp_pdf = now_df_list[m][[sample_time, unit, x_col, y_col] + context_col].reset_index(drop=True)
                        tmp_pdf["cpu_model"] = 0
                        aim_now_df.append(tmp_pdf)

                    else:
                        tmp_pdf = now_df_list[m][
                            [sample_time, unit, x_col, "aim_%s" % y_col] + context_col].reset_index(drop=True)
                        tmp_pdf.rename(columns={"aim_%s" % y_col: y_col}, inplace=True)
                        if m in aim_model_x:
                            tmp_pdf["cpu_model"] = aim_model_x[m]
                        else:
                            aim_model_v = []
                            for kv in aim_model_x.keys():
                                aim_model_v.append(aim_model_x[kv])
                            aim_model_v.sort()

                            aim_model_x[m] = aim_model_v[-1] + 1
                            save_config(aim_model_x, "model_label.json")

                            tmp_pdf["cpu_model"] = aim_model_x[m]

                        aim_now_df.append(tmp_pdf)

                now_df = pd.concat(aim_now_df, axis=0).reset_index(drop=True)
                now_df = now_df.dropna().reset_index(drop=True)
            else:
                aim_now_df = []

                for m in now_df_list.keys():
                    if m == base_model:
                        tmp_pdf = now_df_list[m][[sample_time, unit, x_col, y_col] + context_col].reset_index(drop=True)
                        tmp_pdf["cpu_model"] = 0
                        aim_now_df.append(tmp_pdf)

                    else:
                        tmp_pdf = now_df_list[m][
                            [sample_time, unit, x_col, "aim_%s" % y_col] + context_col].reset_index(drop=True)
                        tmp_pdf.rename(columns={"aim_%s" % y_col: y_col}, inplace=True)
                        if m in aim_model_x:
                            tmp_pdf["cpu_model"] = aim_model_x[m]
                        else:
                            aim_model_v = []
                            for kv in aim_model_x.keys():
                                aim_model_v.append(aim_model_x[kv])
                            aim_model_v.sort()

                            aim_model_x[m] = aim_model_v[-1] + 1
                            save_config(aim_model_x, "model_label.json")

                            tmp_pdf["cpu_model"] = aim_model_x[m]

                        aim_now_df.append(tmp_pdf)

                now_df = pd.concat(aim_now_df, axis=0).reset_index(drop=True)
                now_df = now_df.dropna().reset_index(drop=True)

            print(i)
            print(unit_range[i])
            print(now_df[sample_time].min())
            print(now_df[sample_time].max())
            print(now_df.shape)

            total_corr['start'].append(now_df[sample_time].min())
            total_corr["end"].append(now_df[sample_time].max())
            total_corr['qps_usage_corr'].append(now_df[x_col].corr(now_df[y_col]))
            total_corr['step'].append(i)

            if norm_mode > 1 and norm_mode < 3:
                now_df[x_col] = (now_df[x_col] - base_mean_std["x_mean"]) / base_mean_std["x_std"]
                now_df[y_col] = (now_df[y_col] - base_mean_std["y_mean"]) / base_mean_std["y_std"]
            elif norm_mode >= 3:
                now_df[x_col] = (now_df[x_col] - base_mean_std["x_min"]) / (
                            base_mean_std["x_max"] - base_mean_std["x_min"])
                now_df[y_col] = (now_df[y_col] - base_mean_std["y_min"]) / (
                            base_mean_std["y_max"] - base_mean_std["y_min"])

            # now_df["step"] = i
            df_list_data[i] = now_df

            if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                break
        now_df_k = df_list_data.keys()
        now_df_k = list(now_df_k)
        now_df_k.sort()

        if heter_context:
            context_col = context_col + ["cpu_model"]

        context_col = list(set(context_col))

        total_corr = pd.DataFrame(total_corr)

        if model == "LSDD":
            print("model is the LSDD method")
            if if_update_ref:
                cd_m = LSDDDrift(x_ref=df_list_data[0][[x_col, y_col]].to_numpy(),
                                 backend=backend, p_val=p_val, update_x_ref={update_x_ref: df_list_data[0].shape[0]},
                                 preprocess_fn=None, sigma=None, x_ref_preprocessed=x_ref_preprocessed,
                                 preprocess_at_init=preprocess_at_init,
                                 n_permutations=n_permutations,
                                 n_kernel_centers=None,
                                 lambda_rd_max=lambda_rd_max,
                                 device=device)
                # input_shape: Optional[tuple] = None,
                #             data_type: Optional[str] = None)
            else:
                # , update_x_ref={update_x_ref: init_size}
                cd_m = LSDDDrift(x_ref=df_list_data[0][[x_col, y_col]].to_numpy(),
                                 backend=backend, p_val=p_val, x_ref_preprocessed=x_ref_preprocessed,
                                 preprocess_at_init=preprocess_at_init,
                                 preprocess_fn=None, sigma=None,
                                 n_permutations=n_permutations,
                                 n_kernel_centers=None,
                                 lambda_rd_max=lambda_rd_max,
                                 device=device)
        elif model == "MMD":
            print("model is the MMD method")
            if if_update_ref:
                cd_m = MMDDrift(x_ref=df_list_data[0][[x_col, y_col]].to_numpy(),
                                backend=backend, x_ref_preprocessed=x_ref_preprocessed,
                                preprocess_at_init=preprocess_at_init,
                                p_val=p_val,
                                update_x_ref={update_x_ref: df_list_data[0].shape[0]},
                                preprocess_fn=None,
                                n_permutations=n_permutations,
                                device=device)
            else:
                cd_m = MMDDrift(x_ref=df_list_data[0][[x_col, y_col]].to_numpy(),
                                backend=backend,
                                p_val=p_val, x_ref_preprocessed=x_ref_preprocessed,
                                preprocess_at_init=preprocess_at_init,
                                preprocess_fn=None,
                                n_permutations=n_permutations,
                                device=device)
        else:
            print("model is the Context-aware MMD method")
            if if_update_ref:
                cd_m = ContextMMDDrift(x_ref=df_list_data[0][[x_col, y_col]].astype(np.float32).to_numpy(),
                                       c_ref=df_list_data[0][context_col].astype(np.float32).to_numpy(),
                                       backend=backend, x_ref_preprocessed=x_ref_preprocessed,
                                       preprocess_at_init=preprocess_at_init,
                                       p_val=p_val,
                                       update_ref={update_ref: df_list_data[0].shape[0]},
                                       n_permutations=n_permutations,
                                       prop_c_held=prop_c_held,
                                       n_folds=n_folds,
                                       device=device)
            else:
                cd_m = ContextMMDDrift(x_ref=df_list_data[0][[x_col, y_col]].astype(np.float32).to_numpy(),
                                       c_ref=df_list_data[0][context_col].astype(np.float32).to_numpy(),
                                       backend=backend, x_ref_preprocessed=x_ref_preprocessed,
                                       preprocess_at_init=preprocess_at_init,
                                       p_val=p_val,
                                       n_permutations=n_permutations,
                                       prop_c_held=prop_c_held,
                                       n_folds=n_folds,
                                       device=device)

        print("use gmm metrics are:")
        print(df_list_data[0].columns)
        print("drift model type:")
        print(type(cd_m))
        now_df_list_k = list(df_list_data.keys())
        now_df_list_k.sort()
        print(now_df_list_k)
        #  print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        #             print(f'p-value: {preds["data"]["p_val"]:.3f}')
        #             print(f'p-value threshold: {preds["data"]["threshold"]:.3f}')
        #             print(f'MMD distance: {preds["data"]["distance"]:.3f}')
        #             print(f'MMD distance threshold: {preds["data"]["distance_threshold"]:.3f}')

        # coupling_xx: coupling matrix
        #  for the reference data.
        #
        # coupling_yy: coupling matrix
        #  for the test data.
        #
        # coupling_xy: coupling matrix
        #  between the reference and test data.
        # if model != "LSDD" and model != "MMD":
        #     tmp_res[""] = []
        #     tmp_res[""] = []
        tmp_res = {"label": [], "p-value": [], "p-value-thres": [], "distance": [], "distance-thres": [], "step": []}
        print(context_col)
        for jk in now_df_list_k:
            print(jk)
            tt = time.time()
            if model != "LSDD" and model != "MMD":
                pred_res = cd_m.predict(df_list_data[jk][[x_col, y_col]].astype(np.float32).to_numpy(),
                                        c=df_list_data[jk][context_col].astype(np.float32).to_numpy(),
                                        return_p_val=True, return_distance=True)
            else:
                pred_res = cd_m.predict(df_list_data[jk][[x_col, y_col]].to_numpy(), return_p_val=True,
                                        return_distance=True)
            print(time.time() - tt)
            tmp_res["label"].append(pred_res['data']['is_drift'])
            tmp_res["p-value"].append(pred_res["data"]["p_val"])
            tmp_res["p-value-thres"].append(pred_res['data']['threshold'])
            tmp_res["distance"].append(pred_res["data"]["distance"])
            tmp_res["distance-thres"].append(pred_res["data"]["distance_threshold"])
            tmp_res["step"].append(jk)

            if pred_res["data"]["is_drift"] > 0:
                detected_drifts += 1
            else:
                detected_drifts = 0

            if detected_drifts >= diff_win and early_stop:
                break

        tmp_res = pd.DataFrame(tmp_res)
        total_corr_pdf = pd.merge(total_corr, tmp_res, how="left", on="step")

        # 获取全部与初始训练集起始时刻相同类别的起始时刻条目；

        aim_corr_pdf = total_corr_pdf[total_corr_pdf.label == 0]  # 筛选与初始数据集类别相同的起始时刻条目

        # 为了防止模型过拟合，在符合要求的起始时刻条目内选择最早的起始时间（即选择满足要求的最大数据量）
        # start_time = aim_corr_pdf.start.min()

        # aim_time = aim_corr_pdf.start.unique().tolist()
        aim_corr_pdf = aim_corr_pdf.sort_values("start", ascending=False).reset_index(drop=True)
        total_corr_pdf_list[base_model] = total_corr_pdf
        aim_corr_pdf_list[base_model] = aim_corr_pdf
        aim_times[base_model] = aim_corr_pdf[["start", "end"]].reset_index(drop=True)
        # aim_time = []
        # for j in range(aim_corr_pdf.shape[0]):
        #     aim_time.append({'start':aim_corr_pdf.start.values[j],"end":aim_corr_pdf.end.values[j]})


    else:
        total_corr_pdf_list = {}
        aim_corr_pdf_list = {}
        aim_times = {}
        bic_aic_results_list = {}
        detected_drifts = 0

        # usage_data_list = {}
        for m in x_df_dict.keys():
            print(m)
            now_df_list = {}
            in_df = x_df_dict[m].copy()
            in_df = in_df.sort_values(sample_time).reset_index(drop=True)
            in_df[x_col] = in_df[x_col] * x_unit
            in_df[y_col] = in_df[y_col] * y_unit

            base_mean_std = {"x_mean": in_df[x_col].mean(), "x_std": in_df[x_col].std(),
                             "y_mean": in_df[y_col].mean(), "y_std": in_df[y_col].std(),
                             "x_max": in_df[x_col].max(), "x_min": in_df[x_col].min(),
                             "y_max": in_df[y_col].max(), "y_min": in_df[y_col].min()
                             }

            total_corr = {'start': [], "end": [], 'qps_usage_corr': [], 'step': []}

            unit_range = in_df[unit].unique().tolist()

            unit_range.sort(reverse=inverse)

            print(unit_range)
            for i in range(0, len(unit_range), step):
                if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    now_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                    print(i)
                else:
                    if i == 0:
                        now_df = in_df[(in_df[unit] <= unit_range[i])
                                       & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                    else:
                        now_df = in_df[(in_df[unit] <= unit_range[i])
                                       & (in_df[unit] > unit_range[i + block_size])].reset_index(drop=True)
                    # else:
                    #     if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    #         break
                    #     else:
                    #         continue

                print(i)
                print(unit_range[i])
                print(now_df[sample_time].min())
                print(now_df[sample_time].max())
                print(now_df.shape)

                total_corr['start'].append(now_df[sample_time].min())
                total_corr["end"].append(now_df[sample_time].max())
                total_corr['qps_usage_corr'].append(now_df[x_col].corr(now_df[y_col]))
                total_corr["step"].append(i)

                if norm_mode > 1 and norm_mode < 3:
                    now_df[x_col] = (now_df[x_col] - base_mean_std["x_mean"]) / base_mean_std["x_std"]
                    now_df[y_col] = (now_df[y_col] - base_mean_std["y_mean"]) / base_mean_std["y_std"]
                elif norm_mode >= 3:
                    now_df[x_col] = (now_df[x_col] - base_mean_std["x_min"]) / (
                            base_mean_std["x_max"] - base_mean_std["x_min"])
                    now_df[y_col] = (now_df[y_col] - base_mean_std["y_min"]) / (
                            base_mean_std["y_max"] - base_mean_std["y_min"])

                now_df_list[i] = now_df

                if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    break

            now_df_k = now_df_list.keys()
            now_df_k = list(now_df_k)
            now_df_k.sort()

            total_corr = pd.DataFrame(total_corr)

            if model == "LSDD":
                if if_update_ref:
                    cd_m = LSDDDrift(x_ref=now_df_list[0][[x_col, y_col]].to_numpy(),
                                     backend=backend, p_val=p_val, update_x_ref={update_x_ref: now_df_list[0].shape[0]},
                                     preprocess_fn=None, sigma=None, x_ref_preprocessed=x_ref_preprocessed,
                                     preprocess_at_init=preprocess_at_init,
                                     n_permutations=n_permutations,
                                     n_kernel_centers=None,
                                     lambda_rd_max=lambda_rd_max,
                                     device=device)
                    # input_shape: Optional[tuple] = None,
                    #             data_type: Optional[str] = None)
                else:
                    # , update_x_ref={update_x_ref: init_size}
                    cd_m = LSDDDrift(x_ref=now_df_list[0][[x_col, y_col]].to_numpy(),
                                     backend=backend, p_val=p_val, x_ref_preprocessed=x_ref_preprocessed,
                                     preprocess_at_init=preprocess_at_init,
                                     preprocess_fn=None, sigma=None,
                                     n_permutations=n_permutations,
                                     n_kernel_centers=None,
                                     lambda_rd_max=lambda_rd_max,
                                     device=device)
            elif model == "MMD":
                if if_update_ref:
                    cd_m = MMDDrift(x_ref=now_df_list[0][[x_col, y_col]].to_numpy(),
                                    backend=backend, x_ref_preprocessed=x_ref_preprocessed,
                                    preprocess_at_init=preprocess_at_init,
                                    p_val=p_val,
                                    update_x_ref={update_x_ref: now_df_list[0].shape[0]},
                                    preprocess_fn=None,
                                    n_permutations=n_permutations,
                                    device=device)
                else:
                    cd_m = MMDDrift(x_ref=now_df_list[0][[x_col, y_col]].to_numpy(),
                                    backend=backend,
                                    p_val=p_val, x_ref_preprocessed=x_ref_preprocessed,
                                    preprocess_at_init=preprocess_at_init,
                                    preprocess_fn=None,
                                    n_permutations=n_permutations,
                                    device=device)
            else:
                if if_update_ref:
                    cd_m = ContextMMDDrift(x_ref=now_df_list[0][[x_col, y_col]].astype(np.float32).to_numpy(),
                                           c_ref=now_df_list[0][context_col].astype(np.float32).to_numpy(),
                                           backend=backend, x_ref_preprocessed=x_ref_preprocessed,
                                           preprocess_at_init=preprocess_at_init,
                                           p_val=p_val,
                                           update_ref={update_ref: now_df_list[0].shape[0]},
                                           n_permutations=n_permutations,
                                           prop_c_held=prop_c_held,
                                           n_folds=n_folds,
                                           device=device)
                else:
                    cd_m = ContextMMDDrift(x_ref=now_df_list[0][[x_col, y_col]].astype(np.float32).to_numpy(),
                                           c_ref=now_df_list[0][context_col].astype(np.float32).to_numpy(),
                                           backend=backend, x_ref_preprocessed=x_ref_preprocessed,
                                           preprocess_at_init=preprocess_at_init,
                                           p_val=p_val,
                                           n_permutations=n_permutations,
                                           prop_c_held=prop_c_held,
                                           n_folds=n_folds,
                                           device=device)

            print("use gmm metrics are:")
            print(now_df_list[0].columns)

            now_df_list_k = list(now_df_list.keys())
            now_df_list_k.sort()

            tmp_res = {"label": [], "p-value": [], "p-value-thres": [], "distance": [], "distance-thres": [],
                       "step": []}
            for jk in now_df_list_k:
                if model != "LSDD" and model != "MMD":
                    pred_res = cd_m.predict(now_df_list[jk][[x_col, y_col]].astype(np.float32).to_numpy(),
                                            c=now_df_list[jk][context_col].astype(np.float32).to_numpy(),
                                            return_p_val=True, return_distance=True)
                else:
                    pred_res = cd_m.predict(now_df_list[jk][[x_col, y_col]].to_numpy(), return_p_val=True,
                                            return_distance=True)

                tmp_res["label"].append(pred_res['data']['is_drift'])
                tmp_res["p-value"].append(pred_res["data"]["p_val"])
                tmp_res["p-value-thres"].append(pred_res['data']['threshold'])
                tmp_res["distance"].append(pred_res["data"]["distance"])
                tmp_res["distance-thres"].append(pred_res["data"]["distance_threshold"])
                tmp_res["step"].append(jk)

                if pred_res["data"]["is_drift"] > 0:
                    detected_drifts += 1
                else:
                    detected_drifts = 0

                if detected_drifts >= diff_win and early_stop:
                    break

            tmp_res = pd.DataFrame(tmp_res)
            total_corr_pdf = pd.merge(total_corr, tmp_res, how="left", on="step")

            # 获取全部与初始训练集起始时刻相同类别的起始时刻条目；

            aim_corr_pdf = total_corr_pdf[total_corr_pdf.label == 0].sort_values("start", ascending=False).reset_index(
                drop=True)  # 筛选与初始数据集类别相同的起始时刻条目

            total_corr_pdf_list[m] = total_corr_pdf
            aim_corr_pdf_list[m] = aim_corr_pdf
            aim_times[m] = aim_corr_pdf[["start", "end"]].reset_index(drop=True)

    if save:
        if not os.path.exists(os.path.join(save_path, "detector_tf")):
            os.makedirs(os.path.join(save_path, "detector_tf"), exist_ok=True)
        save_detector(cd_m, filepath=os.path.join(save_path, "detector_tf"))

    # if use_gpu:
    #     torch.cuda.empty_cache()

    return total_corr_pdf_list, aim_corr_pdf_list, aim_times


def find_train_range_gmm(x_df_dict, base_model="8269CY", init_size=1, freq='t', step=1, n_components=None,
                         select_n='bic', diff_limit=0.15,
                         diff_win=3, unit="ds", x_col="qps__Q_s", y_col='cpu_usage', test_size=0.2,
                         sample_time='sample_time__m', norm=False, x_unit=1, y_unit=60, block_size=1,
                         standard=False, lateset_range=2, mode=1, norm_mode=1, cov_col="cpu_util__pct", pdc_col='RUE'):
    '''
           x_df: 输入数据,原始API经过generate_train_data函数处理后生成的数据集
           init_size: 起始数据长度，默认为数据集中最新2天的数据
           freq: 输入init_size的单位，more为t,即间隔为1分钟，可选为：t,h,s,d，表示数据时间间隔分别为1分钟，1小时，1秒，1天
           step: 每次增加的历史数据长度，单位与init_size相同
           n_components: 指定的聚类类别数量，默认为None，若为正整数则不自动探索合适的聚类数量，否则则探索聚类数量
           select_n: 若探索聚类数量，选择哪种指标来进行n_components的选取，默认为bic
           diff_limit: 若探索聚类数量，见find_n_components
           diff_win: 若探索聚类数量，见find_n_components
    '''

    if freq not in ['t', 'h', 's', 'd']:
        print("the freq is invalid: the valid set:['t','h','s','d'];")
        print("t: minute;")
        print("h: hour;")
        print("s: second;")
        print("d: day")

    if standard:
        total_corr_pdf_list = {}
        aim_corr_pdf_list = {}
        aim_times = {}
        bic_aic_results_list = {}
        # usage_data_list = {}

        # perf_alpha_list = {}

        in_df = x_df_dict[base_model].copy()
        in_df = in_df.sort_values(sample_time).reset_index(drop=True)
        in_df[x_col] = in_df[x_col] * x_unit
        in_df[y_col] = in_df[y_col] * y_unit
        total_corr = {'start': [], "end": [], 'qps_usage_corr': [], 'qps_usage_k': [], 'qps_usage_b': [], 'mse': [],
                      'r2score': [],
                      'mape': [], 'step': []}

        unit_range = in_df[unit].unique().tolist()

        unit_range.sort(reverse=True)

        for i in range(0, len(unit_range), step):
            now_df_list = {}
            for m in x_df_dict.keys():
                if m == base_model:
                    if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                        now_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                        print(i)
                    else:
                        if i == 0:
                            now_df = in_df[(in_df[unit] <= unit_range[i])
                                           & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                        else:
                            now_df = in_df[(in_df[unit] <= unit_range[i])
                                           & (in_df[unit] > unit_range[i + block_size])].reset_index(drop=True)

                    now_df_list[m] = now_df
                    now_df_list[m]["map"] = 1.0
                    now_df_list[m]["pred_map"] = 1.0
                    now_df_list[m]["base_%s" % y_col] = now_df_list[m][y_col].copy()
                    now_df_list[m]["aim_%s" % y_col] = now_df_list[m][y_col].copy()
                    now_df_list[m]["base_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                    now_df_list[m]["aim_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                    now_df_list[m]["base_%s" % cov_col] = now_df_list[m][cov_col].copy()
                    now_df_list[m]["base_%s" % x_col] = now_df_list[m][x_col].copy()
                else:
                    if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                        now_base_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                        now_raw_df = x_df_dict[m][x_df_dict[m][unit] <= unit_range[i]].copy().reset_index(drop=True)
                        print(i)
                    else:
                        if i == 0:
                            now_base_df = in_df[(in_df[unit] <= unit_range[i])
                                                & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                            now_raw_df = x_df_dict[m][(x_df_dict[m][unit] <= unit_range[i])
                                                      & (x_df_dict[m][unit] > unit_range[i + init_size])].reset_index(
                                drop=True)
                        else:
                            now_base_df = in_df[(in_df[unit] <= unit_range[i])
                                                & (in_df[unit] > unit_range[i + block_size])].copy().reset_index(
                                drop=True)
                            now_raw_df = x_df_dict[m][(x_df_dict[m][unit] <= unit_range[i])
                                                      & (x_df_dict[m][unit] > unit_range[i + block_size])].reset_index(
                                drop=True)

                    # now_base_df.rename(columns={cov_col:"base_%s" % cov_col},inplace=True)
                    if now_raw_df.shape[0] < 1:
                        continue
                    now_raw_df[x_col] = now_raw_df[x_col] * x_unit
                    now_raw_df[y_col] = now_raw_df[y_col] * y_unit

                    now_base_df["base_%s" % y_col] = now_base_df[y_col].copy()
                    now_base_df["base_%s" % cov_col] = now_base_df[cov_col].copy()
                    now_base_df["base_%s" % pdc_col] = now_base_df[pdc_col].copy()
                    now_base_df["base_%s" % x_col] = now_base_df[x_col].copy()
                    now_raw_df = pd.merge(now_base_df[
                                              [sample_time, "base_%s" % cov_col, "base_%s" % y_col, "base_%s" % pdc_col,
                                               "base_%s" % x_col]], now_raw_df, on=sample_time)
                    if now_raw_df.shape[0] < 1:
                        continue
                    now_raw_df = now_raw_df.sort_values(sample_time)
                    now_base_df = now_base_df.sort_values(sample_time)

                    now_raw_df["map"] = now_raw_df["base_%s" % pdc_col] / now_raw_df[pdc_col]

                    perf_map = UsageMapPredictor()
                    perf_map.train(X_train=now_raw_df[["base_%s" % cov_col]].to_numpy(),
                                   y_train=now_raw_df["map"].to_numpy())

                    maps = perf_map.predict_ll(now_raw_df[["base_%s" % cov_col]].to_numpy())
                    now_raw_df["pred_map"] = pd.Series(maps)
                    now_raw_df["aim_%s" % y_col] = now_raw_df[y_col] * now_raw_df["pred_map"]

                    now_df_list[m] = now_raw_df
                    # print(now_raw_df.head(10))
                    print(MAPE(now_raw_df["base_%s" % y_col],
                               now_raw_df["aim_%s" % y_col] / now_raw_df[x_col] * now_raw_df["base_%s" % x_col]))
                    print(MAPE(now_raw_df["map"], now_raw_df["pred_map"]))

            aim_now_df = []
            for m in now_df_list.keys():
                if m == base_model:
                    aim_now_df.append(now_df_list[m][[sample_time, unit, x_col, y_col]].reset_index(drop=True))
                else:
                    tmp_pdf = now_df_list[m][[sample_time, unit, x_col, "aim_%s" % y_col]].reset_index(drop=True)
                    tmp_pdf.rename(columns={"aim_%s" % y_col: y_col}, inplace=True)
                    aim_now_df.append(tmp_pdf)

            now_df = pd.concat(aim_now_df, axis=0).reset_index(drop=True)

            print(i)
            print(unit_range[i])
            print(now_df[sample_time].min())
            print(now_df[sample_time].max())
            print(now_df.shape)

            total_corr['start'].append(now_df[sample_time].min())
            total_corr["end"].append(now_df[sample_time].max())
            total_corr['qps_usage_corr'].append(now_df[x_col].corr(now_df[y_col]))
            total_corr['step'].append(i)

            tmp_lr = LinearRegression()

            if i == 0:
                tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test, _ = preprocess_qps_usage_data(now_df,
                                                                                                x_column=x_col,
                                                                                                y_column=y_col,
                                                                                                norm=norm,
                                                                                                test_size=test_size)
            else:
                tmp_x_train, _, tmp_y_train, _, _ = preprocess_qps_usage_data(now_df, x_column=x_col,
                                                                              y_column=y_col,
                                                                              norm=norm, test_size=test_size)
            tmp_lr.fit(tmp_x_train, tmp_y_train)
            total_corr['qps_usage_k'].append(tmp_lr.coef_[0][0])
            total_corr['qps_usage_b'].append(tmp_lr.intercept_[0])
            total_corr['mse'].append(MSE(y_true=tmp_y_test, y_pred=tmp_lr.predict(tmp_x_test)))
            total_corr['r2score'].append(tmp_lr.score(tmp_x_test, tmp_y_test))
            total_corr['mape'].append(MAPE(y_true=tmp_y_test, y_pred=tmp_lr.predict(tmp_x_test)))

            if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                break

        total_corr_pdf = pd.DataFrame(total_corr)

        total_corr_pdf = total_corr_pdf.reset_index(drop=True)
        if norm_mode >= 2:
            total_corr_pdf['qps_usage_k_norm'] = (total_corr_pdf['qps_usage_k'] - total_corr_pdf[
                'qps_usage_k'].min()) / (total_corr_pdf['qps_usage_k'].max() - total_corr_pdf['qps_usage_k'].min())
            total_corr_pdf['qps_usage_b_norm'] = (total_corr_pdf['qps_usage_b'] - total_corr_pdf[
                'qps_usage_b'].min()) / (total_corr_pdf['qps_usage_b'].max() - total_corr_pdf['qps_usage_b'].min())
            total_corr_pdf['mape_norm'] = (total_corr_pdf['mape'] - total_corr_pdf['mape'].min()) / (
                    total_corr_pdf['mape'].max() - total_corr_pdf['mape'].min())
            total_corr_pdf['step_norm'] = (total_corr_pdf['step'] - total_corr_pdf['step'].min()) / (
                    total_corr_pdf['step'].max() - total_corr_pdf['step'].min())
        else:
            total_corr_pdf['qps_usage_k_norm'] = (total_corr_pdf['qps_usage_k'] - total_corr_pdf[
                'qps_usage_k'].mean()) / (total_corr_pdf['qps_usage_k'].std())
            total_corr_pdf['qps_usage_b_norm'] = (total_corr_pdf['qps_usage_b'] - total_corr_pdf[
                'qps_usage_b'].mean()) / (total_corr_pdf['qps_usage_b'].std())
            total_corr_pdf['mape_norm'] = (total_corr_pdf['mape'] - total_corr_pdf['mape'].mean()) / (
                total_corr_pdf['mape'].std())
            total_corr_pdf['step_norm'] = (total_corr_pdf['step'] - total_corr_pdf['step'].mean()) / (
                total_corr_pdf['step'].std())

        if mode >= 4:
            if norm_mode > 0:
                X_input_raw = total_corr_pdf[["qps_usage_k_norm", "qps_usage_b_norm"]]
            else:
                X_input_raw = total_corr_pdf[["qps_usage_k", "qps_usage_b"]]

        if mode >= 3 and mode < 4:
            if norm_mode > 0:
                X_input_raw = total_corr_pdf[["qps_usage_k_norm"]]
            else:
                X_input_raw = total_corr_pdf[["qps_usage_k"]]

        elif mode >= 2 and mode < 3:
            if norm_mode > 0:
                X_input_raw = total_corr_pdf[["qps_usage_k_norm", "mape_norm"]]
            else:
                X_input_raw = total_corr_pdf[["qps_usage_k", "mape"]]
        elif mode < 2 and mode >= 1:
            if norm_mode > 0:
                X_input_raw = total_corr_pdf[["qps_usage_k_norm", "qps_usage_b_norm", "mape_norm"]]
            else:
                X_input_raw = total_corr_pdf[["qps_usage_k", "qps_usage_b", "mape"]]
        else:
            if norm_mode > 0:
                X_input_raw = total_corr_pdf[["qps_usage_k_norm", "qps_usage_b_norm", "mape_norm", 'step_norm']]
            else:
                X_input_raw = total_corr_pdf[["qps_usage_k", "qps_usage_b", "mape", 'step']]

        # 原始结果方便展示
        X_input = total_corr_pdf[["qps_usage_k", "qps_usage_b", "mape"]].to_numpy()
        print("use gmm metrics are:")
        print(X_input_raw.columns)
        X_input_raw = X_input_raw.to_numpy()

        # 对提取的feature进行聚类，聚类类别的上限根据BIC算法自动确定：对于从分为2类开始，设定不同类别上限来训练GMM模型，
        # 当训练后模型的BIC指标的相对变化连续较小（默认为连续小于0.15）时，即认为寻找到合适的分类类别，
        # 选用BIC指标是因为相对于AIC，BIC更容易选择简单的模型，即倾向于选择较少的类别数，
        # 因为起始时刻条目有限，所以无需选择较大数量的分类类别数
        if not n_components or n_components < 0:
            # 设置分类上限为数据条目数量，获取模型性能，可以发现根据BIC指标，在5~7类时效果已经稳定：
            # total_corr_pdf.shape[0]
            exp_n_components = np.arange(1, 7)
            print("exp_n_components:")
            print(exp_n_components)
            models = [GMM(n, covariance_type='full', random_state=0).fit(X_input_raw) for n in exp_n_components]

            bic_aic_df_input = pd.DataFrame(
                {'component': exp_n_components.tolist(), 'bic': [m.bic(X_input_raw) for m in models],
                 'aic': [m.aic(X_input_raw) for m in models]})

            print("bic_aic_df_input:")
            print(bic_aic_df_input)
            selected_n_component, bic_aic_results = find_n_components_gmm(bic_aic_pdf=bic_aic_df_input,
                                                                          com_col='component', select_n=select_n,
                                                                          diff_limit=diff_limit, diff_win=diff_win)
            print("find the reasonable n_component: %d" % selected_n_component)

            # 对起始时刻条目按照提取的3个维度和自动探索到的合理分类类别上限来训练GMM模型
            gmm = GMM(n_components=int(selected_n_component)).fit(X_input_raw)
        else:
            # 若给定了聚类中类别数目则不进行类别数目自动检索
            gmm = GMM(n_components=int(np.round(n_components))).fit(X_input_raw)

        # 获取聚类结果，表达为分类标签（label）
        label_input = gmm.predict(X_input_raw)

        # 获取全部与初始训练集起始时刻相同类别的起始时刻条目；
        total_corr_pdf = total_corr_pdf.reset_index(drop=True)
        total_corr_pdf['label'] = pd.Series(label_input.tolist())
        total_corr_pdf = total_corr_pdf.sort_values("start", ascending=False).reset_index(drop=True)
        least_label = total_corr_pdf['label'][0]  # 获取初始训练集label
        aim_corr_pdf = total_corr_pdf[total_corr_pdf.label == least_label]  # 筛选与初始数据集类别相同的起始时刻条目

        # 为了防止模型过拟合，在符合要求的起始时刻条目内选择最早的起始时间（即选择满足要求的最大数据量）
        # start_time = aim_corr_pdf.start.min()

        # aim_time = aim_corr_pdf.start.unique().tolist()
        aim_corr_pdf = aim_corr_pdf.sort_values("start", ascending=False).reset_index(drop=True)
        # aim_time = []
        # for j in range(aim_corr_pdf.shape[0]):
        #     aim_time.append({'start':aim_corr_pdf.start.values[j],"end":aim_corr_pdf.end.values[j]})

        if not n_components or n_components < 0:
            total_corr_pdf_list[base_model] = total_corr_pdf
            aim_corr_pdf_list[base_model] = aim_corr_pdf
            aim_times[base_model] = aim_corr_pdf[["start", "end"]].reset_index(drop=True)
            bic_aic_results_list[base_model] = bic_aic_results
            # for m in usage_data_list.keys():
            #     usage_data_list[m] = usage_data_list[m][usage_data_list[m][unit].isin(aim_time)].copy().reset_index(
            #         drop=True)
            # return total_corr_pdf, aim_corr_pdf, aim_time,
        else:
            total_corr_pdf_list[base_model] = total_corr_pdf
            aim_corr_pdf_list[base_model] = aim_corr_pdf
            aim_times[base_model] = aim_corr_pdf[["start", "end"]].reset_index(drop=True)
            bic_aic_results_list[base_model] = None
            # for m in usage_data_list.keys():
            #     usage_data_list[m] = usage_data_list[m][usage_data_list[m][unit].isin(aim_time)].copy().reset_index(
            #         drop=True)

    else:
        total_corr_pdf_list = {}
        aim_corr_pdf_list = {}
        aim_times = {}
        bic_aic_results_list = {}
        # usage_data_list = {}
        for m in x_df_dict.keys():
            print(m)
            in_df = x_df_dict[m].copy()
            in_df = in_df.sort_values(sample_time).reset_index(drop=True)
            in_df[x_col] = in_df[x_col] * x_unit
            in_df[y_col] = in_df[y_col] * y_unit
            total_corr = {'start': [], "end": [], 'qps_usage_corr': [], 'qps_usage_k': [], 'qps_usage_b': [], 'mse': [],
                          'r2score': [],
                          'mape': []}

            unit_range = in_df[unit].unique().tolist()

            unit_range.sort(reverse=True)

            print(unit_range)
            for i in range(0, len(unit_range), step):
                if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    now_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                    print(i)
                else:
                    if i == 0:
                        now_df = in_df[(in_df[unit] <= unit_range[i])
                                       & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                    else:
                        now_df = in_df[(in_df[unit] <= unit_range[i])
                                       & (in_df[unit] > unit_range[i + block_size])].reset_index(drop=True)
                    # else:
                    #     if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    #         break
                    #     else:
                    #         continue

                print(i)
                print(unit_range[i])
                print(now_df[sample_time].min())
                print(now_df[sample_time].max())
                print(now_df.shape)

                total_corr['start'].append(now_df[sample_time].min())
                total_corr["end"].append(now_df[sample_time].max())
                total_corr['qps_usage_corr'].append(now_df[x_col].corr(now_df[y_col]))

                tmp_lr = LinearRegression()

                if i == 0:
                    tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test, _ = preprocess_qps_usage_data(now_df,
                                                                                                    x_column=x_col,
                                                                                                    y_column=y_col,
                                                                                                    norm=norm,
                                                                                                    test_size=test_size)
                else:
                    tmp_x_train, _, tmp_y_train, _, _ = preprocess_qps_usage_data(now_df, x_column=x_col,
                                                                                  y_column=y_col,
                                                                                  norm=norm, test_size=test_size)
                tmp_lr.fit(tmp_x_train, tmp_y_train)
                total_corr['qps_usage_k'].append(tmp_lr.coef_[0][0])
                total_corr['qps_usage_b'].append(tmp_lr.intercept_[0])
                total_corr['mse'].append(MSE(y_true=tmp_y_test, y_pred=tmp_lr.predict(tmp_x_test)))
                total_corr['r2score'].append(tmp_lr.score(tmp_x_test, tmp_y_test))
                total_corr['mape'].append(MAPE(y_true=tmp_y_test, y_pred=tmp_lr.predict(tmp_x_test)))

                if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    break

            total_corr_pdf = pd.DataFrame(total_corr)

            total_corr_pdf = total_corr_pdf.reset_index(drop=True)

            if norm_mode >= 2:
                total_corr_pdf['qps_usage_k_norm'] = (total_corr_pdf['qps_usage_k'] - total_corr_pdf[
                    'qps_usage_k'].min()) / (total_corr_pdf['qps_usage_k'].max() - total_corr_pdf['qps_usage_k'].min())
                total_corr_pdf['qps_usage_b_norm'] = (total_corr_pdf['qps_usage_b'] - total_corr_pdf[
                    'qps_usage_b'].min()) / (total_corr_pdf['qps_usage_b'].max() - total_corr_pdf['qps_usage_b'].min())
                total_corr_pdf['mape_norm'] = (total_corr_pdf['mape'] - total_corr_pdf['mape'].min()) / (
                        total_corr_pdf['mape'].max() - total_corr_pdf['mape'].min())
            else:
                total_corr_pdf['qps_usage_k_norm'] = (total_corr_pdf['qps_usage_k'] - total_corr_pdf[
                    'qps_usage_k'].mean()) / (total_corr_pdf['qps_usage_k'].std())
                total_corr_pdf['qps_usage_b_norm'] = (total_corr_pdf['qps_usage_b'] - total_corr_pdf[
                    'qps_usage_b'].mean()) / (total_corr_pdf['qps_usage_b'].std())
                total_corr_pdf['mape_norm'] = (total_corr_pdf['mape'] - total_corr_pdf['mape'].mean()) / (
                    total_corr_pdf['mape'].std())

            if mode > 0:
                if norm_mode > 0:
                    X_input_raw = total_corr_pdf[["qps_usage_k_norm", "mape_norm"]].to_numpy()
                else:
                    X_input_raw = total_corr_pdf[["qps_usage_k", "mape"]].to_numpy()
            else:
                if norm_mode > 0:
                    X_input_raw = total_corr_pdf[["qps_usage_k_norm", "qps_usage_b_norm", "mape_norm"]].to_numpy()
                else:
                    X_input_raw = total_corr_pdf[["qps_usage_k", "qps_usage_b", "mape"]].to_numpy()

            #             "qps_usage_b_norm",
            # 原始结果方便展示
            X_input = total_corr_pdf[["qps_usage_k", "qps_usage_b", "mape"]].to_numpy()
            #         "qps_usage_b",

            # 对提取的feature进行聚类，聚类类别的上限根据BIC算法自动确定：对于从分为2类开始，设定不同类别上限来训练GMM模型，
            # 当训练后模型的BIC指标的相对变化连续较小（默认为连续小于0.15）时，即认为寻找到合适的分类类别，
            # 选用BIC指标是因为相对于AIC，BIC更容易选择简单的模型，即倾向于选择较少的类别数，
            # 因为起始时刻条目有限，所以无需选择较大数量的分类类别数
            if not n_components or n_components < 0:
                # 设置分类上限为数据条目数量，获取模型性能，可以发现根据BIC指标，在5~7类时效果已经稳定：
                exp_n_components = np.arange(1, total_corr_pdf.shape[0])
                models = [GMM(n, covariance_type='full', random_state=0).fit(X_input_raw) for n in exp_n_components]

                bic_aic_df_input = pd.DataFrame(
                    {'component': exp_n_components.tolist(), 'bic': [m.bic(X_input_raw) for m in models],
                     'aic': [m.aic(X_input_raw) for m in models]})
                selected_n_component, bic_aic_results = find_n_components_gmm(bic_aic_pdf=bic_aic_df_input,
                                                                              com_col='component', select_n=select_n,
                                                                              diff_limit=diff_limit, diff_win=diff_win)
                print("find the reasonable n_component: %d" % selected_n_component)

                # 对起始时刻条目按照提取的3个维度和自动探索到的合理分类类别上限来训练GMM模型
                gmm = GMM(n_components=int(selected_n_component)).fit(X_input_raw)
            else:
                # 若给定了聚类中类别数目则不进行类别数目自动检索
                gmm = GMM(n_components=int(np.round(n_components))).fit(X_input_raw)

            # 获取聚类结果，表达为分类标签（label）
            label_input = gmm.predict(X_input_raw)

            # 获取全部与初始训练集起始时刻相同类别的起始时刻条目；
            total_corr_pdf = total_corr_pdf.reset_index(drop=True)
            total_corr_pdf['label'] = pd.Series(label_input.tolist())
            total_corr_pdf = total_corr_pdf.sort_values("start", ascending=False).reset_index(drop=True)
            least_label = total_corr_pdf['label'][0]  # 获取初始训练集label
            aim_corr_pdf = total_corr_pdf[total_corr_pdf.label == least_label]  # 筛选与初始数据集类别相同的起始时刻条目

            aim_corr_pdf = aim_corr_pdf.sort_values("start", ascending=False).reset_index(drop=True)

            # 为了防止模型过拟合，在符合要求的起始时刻条目内选择最早的起始时间（即选择满足要求的最大数据量）
            # start_time = aim_corr_pdf.start.min()
            # aim_time = aim_corr_pdf.start.unique().tolist()

            if not n_components or n_components < 0:
                total_corr_pdf_list[m] = total_corr_pdf
                aim_corr_pdf_list[m] = aim_corr_pdf
                aim_times[m] = aim_corr_pdf[["start", "end"]].reset_index(drop=True)
                bic_aic_results_list[m] = bic_aic_results
                # usage_data_list[m] = x_df_dict[m][x_df_dict[m][unit].isin(aim_time)].copy().reset_index(drop=True)
                # return total_corr_pdf, aim_corr_pdf, aim_time,
            else:
                total_corr_pdf_list[m] = total_corr_pdf
                aim_corr_pdf_list[m] = aim_corr_pdf
                aim_times[m] = aim_corr_pdf[["start", "end"]].reset_index(drop=True)
                bic_aic_results_list[m] = None
                # usage_data_list[m] = x_df_dict[m][x_df_dict[m][unit].isin(aim_time)].copy().reset_index(drop=True)

    return total_corr_pdf_list, aim_corr_pdf_list, aim_times, bic_aic_results_list


def find_train_range_threshold(x_df_dict, base_model="8269CY", init_size=1, step=1,
                               threshold_limit=0.15,
                               threshold_win=3, unit="hour", x_col="qps__Q_s", y_col='cpu_usage', test_size=0.1,
                               sample_time='sample_time_n', norm=False, x_unit=1, y_unit=1000,
                               block_size=1, cov_col="cpu_util__pct", pdc_col='RUE',
                               threshold_col='qps_usage_k', find_mode=0, heter_update=False):
    '''
           x_df: 输入数据,原始API经过generate_train_data函数处理后生成的数据集
           init_size: 起始数据长度，默认为数据集中最新2天的数据
           freq: 输入init_size的单位，more为t,即间隔为1分钟，可选为：t,h,s,d，表示数据时间间隔分别为1分钟，1小时，1秒，1天
           step: 每次增加的历史数据长度，单位与init_size相同
           n_components: 指定的聚类类别数量，默认为None，若为正整数则不自动探索合适的聚类数量，否则则探索聚类数量
           select_n: 若探索聚类数量，选择哪种指标来进行n_components的选取，默认为bic
           diff_limit: 若探索聚类数量，见find_n_components
           diff_win: 若探索聚类数量，见find_n_components
    '''

    # standard=False
    total_corr_pdf_list = {}
    aim_corr_pdf_list = {}
    aim_times = {}
    bic_aic_results_list = {}

    in_df = x_df_dict[base_model].copy()
    in_df[sample_time] = pd.to_datetime(in_df[sample_time])
    in_df = in_df.sort_values(sample_time).reset_index(drop=True)
    in_df[x_col] = in_df[x_col] * x_unit
    in_df[y_col] = in_df[y_col] * y_unit
    total_corr = {'start': [], "end": [], 'qps_usage_corr': [], 'qps_usage_k': [], 'qps_usage_b': [], 'mse': [],
                  'r2score': [],
                  'mape': [], 'step': [], 'rue': []}

    unit_range = in_df[unit].unique().tolist()

    unit_range.sort(reverse=True)

    now_patience = 0

    time_range = []

    get_test = False

    for i in range(0, len(unit_range), step):
        now_df_list = {}
        for m in x_df_dict.keys():
            if m == base_model:
                if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    now_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                    print(i)
                else:
                    if i == 0:
                        now_df = in_df[(in_df[unit] <= unit_range[i])
                                       & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                    else:
                        now_df = in_df[(in_df[unit] <= unit_range[i])
                                       & (in_df[unit] > unit_range[i + block_size])].reset_index(drop=True)

                now_df_list[m] = now_df
                now_df_list[m]["map"] = 1.0
                now_df_list[m]["pred_map"] = 1.0
                now_df_list[m]["base_%s" % y_col] = now_df_list[m][y_col].copy()
                now_df_list[m]["aim_%s" % y_col] = now_df_list[m][y_col].copy()
                now_df_list[m]["base_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                now_df_list[m]["aim_%s" % pdc_col] = now_df_list[m][pdc_col].copy()
                now_df_list[m]["base_%s" % cov_col] = now_df_list[m][cov_col].copy()
                now_df_list[m]["base_%s" % x_col] = now_df_list[m][x_col].copy()
            else:
                if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
                    now_base_df = in_df[in_df[unit] <= unit_range[i]].reset_index(drop=True)
                    now_raw_df = x_df_dict[m][x_df_dict[m][unit] <= unit_range[i]].copy().reset_index(drop=True)
                    now_raw_df[sample_time] = pd.to_datetime(now_raw_df[sample_time])

                    print(i)
                else:
                    if i == 0:
                        now_base_df = in_df[(in_df[unit] <= unit_range[i])
                                            & (in_df[unit] > unit_range[i + init_size])].reset_index(drop=True)
                        now_raw_df = x_df_dict[m][(x_df_dict[m][unit] <= unit_range[i])
                                                  & (x_df_dict[m][unit] > unit_range[i + init_size])].reset_index(
                            drop=True)
                        now_raw_df[sample_time] = pd.to_datetime(now_raw_df[sample_time])
                    else:
                        now_base_df = in_df[(in_df[unit] <= unit_range[i])
                                            & (in_df[unit] > unit_range[i + block_size])].copy().reset_index(
                            drop=True)
                        now_raw_df = x_df_dict[m][(x_df_dict[m][unit] <= unit_range[i])
                                                  & (x_df_dict[m][unit] > unit_range[i + block_size])].reset_index(
                            drop=True)
                        now_raw_df[sample_time] = pd.to_datetime(now_raw_df[sample_time])

                # now_base_df.rename(columns={cov_col:"base_%s" % cov_col},inplace=True)
                if now_raw_df.shape[0] < 1:
                    continue
                if now_base_df.shape[0] < 1:
                    continue

                now_raw_df[x_col] = now_raw_df[x_col] * x_unit
                now_raw_df[y_col] = now_raw_df[y_col] * y_unit

                now_base_df["base_%s" % y_col] = now_base_df[y_col].copy()
                now_base_df["base_%s" % cov_col] = now_base_df[cov_col].copy()
                now_base_df["base_%s" % pdc_col] = now_base_df[pdc_col].copy()
                now_base_df["base_%s" % x_col] = now_base_df[x_col].copy()

                now_raw_df = pd.merge(now_base_df[
                                          [sample_time, "base_%s" % cov_col, "base_%s" % y_col, "base_%s" % pdc_col,
                                           "base_%s" % x_col]], now_raw_df, on=sample_time)
                if now_raw_df.shape[0] < 1:
                    continue

                now_raw_df = now_raw_df.sort_values(sample_time)
                # now_base_df = now_base_df.sort_values(sample_time)
                if heter_update:
                    now_raw_df["map"] = now_raw_df["base_%s" % pdc_col] / now_raw_df[pdc_col]

                    perf_map = UsageMapPredictor()
                    perf_map.train(X_train=now_raw_df[["base_%s" % cov_col]].to_numpy(),
                                   y_train=now_raw_df["map"].to_numpy())

                    maps = perf_map.predict_ll(now_raw_df[["base_%s" % cov_col]].to_numpy())
                    now_raw_df["pred_map"] = pd.Series(maps)
                    now_raw_df["aim_%s" % y_col] = now_raw_df[y_col] * now_raw_df["pred_map"]

                    now_df_list[m] = now_raw_df
                    # print(now_raw_df.head(10))
                    print(MAPE(now_raw_df["base_%s" % y_col],
                               now_raw_df["aim_%s" % y_col] / now_raw_df[x_col] * now_raw_df["base_%s" % x_col]))
                    print(MAPE(now_raw_df["map"], now_raw_df["pred_map"]))
                else:
                    now_raw_df["aim_%s" % y_col] = now_raw_df[y_col].copy()

        aim_now_df = []

        for m in now_df_list.keys():
            if m == base_model:
                aim_now_df.append(now_df_list[m][[sample_time, unit, x_col, y_col]].reset_index(drop=True))
            else:
                tmp_pdf = now_df_list[m][[sample_time, unit, x_col, "aim_%s" % y_col]].reset_index(drop=True)
                tmp_pdf.rename(columns={"aim_%s" % y_col: y_col}, inplace=True)
                aim_now_df.append(tmp_pdf)

        now_df = pd.concat(aim_now_df, axis=0).reset_index(drop=True)
        now_df = now_df.dropna().reset_index(drop=True)

        if now_df.shape[0] < 0:
            continue

        print(i)
        print(unit_range[i])
        print(now_df[sample_time].min())
        print(now_df[sample_time].max())
        print(now_df.shape)

        total_corr['start'].append(now_df[sample_time].min())
        total_corr["end"].append(now_df[sample_time].max())
        total_corr['qps_usage_corr'].append(now_df[x_col].corr(now_df[y_col]))
        total_corr['step'].append(i)

        tmp_lr = LinearRegression()

        if not get_test:
            tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test, _ = preprocess_qps_usage_data(now_df,
                                                                                            x_column=x_col,
                                                                                            y_column=y_col,
                                                                                            norm=norm,
                                                                                            test_size=test_size)
            get_test = True
        else:
            tmp_x_train, _, tmp_y_train, _, _ = preprocess_qps_usage_data(now_df, x_column=x_col,
                                                                          y_column=y_col,
                                                                          norm=norm, test_size=test_size)

        tmp_lr.fit(tmp_x_train, tmp_y_train)
        total_corr['qps_usage_k'].append(tmp_lr.coef_[0][0])
        total_corr['rue'].append((now_df[y_col] / now_df[x_col]).mean())
        total_corr['qps_usage_b'].append(tmp_lr.intercept_[0])
        total_corr['mse'].append(MSE(y_true=tmp_y_test, y_pred=tmp_lr.predict(tmp_x_test)))
        total_corr['r2score'].append(tmp_lr.score(tmp_x_test, tmp_y_test))
        total_corr['mape'].append(MAPE(y_true=tmp_y_test, y_pred=tmp_lr.predict(tmp_x_test)))

        # time_range

        if find_mode < 1:
            if i > 0:
                diffs = np.abs(total_corr[threshold_col][-1] - total_corr[threshold_col][0]) / np.abs(
                    total_corr[threshold_col][0])
                print("diffs is: %f latest k: %f now k: %f" % (diffs, total_corr[threshold_col][0],
                                                               total_corr[threshold_col][-1]))
                if diffs > threshold_limit:
                    now_patience += 1
                else:
                    time_range += (now_df[unit].unique().tolist())

                if now_patience >= threshold_win:
                    break
            else:
                time_range += (now_df[unit].unique().tolist())
        else:
            if i > 0:
                diffs = np.abs(total_corr[threshold_col][-1] - total_corr[threshold_col][0]) / np.abs(
                    total_corr[threshold_col][0])
                print("diffs is: %f latest k: %f now k: %f" % (diffs, total_corr[threshold_col][0],
                                                               total_corr[threshold_col][-1]))
                # print(diffs)
                if diffs > threshold_limit:
                    now_patience += 1
                else:
                    time_range += (now_df[unit].unique().tolist())
            else:
                time_range += (now_df[unit].unique().tolist())

        if (i + init_size) >= len(unit_range) or (i + block_size) >= len(unit_range):
            break

    return list(set(time_range))




