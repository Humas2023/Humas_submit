import os
import numpy as np
import pandas as pd

import utils
from usage_predict import PredictUsageModel
from heter_predict_model import UsageMapPredictor

import argparse
import time
import json
# from find_range import find_train_range_gmm
from find_range import find_train_range_threshold
# from heter_predict_model import UsageMapPredictor
import joblib

http = 'HTTP'
hsf = 'HSF'
hsf_consumer = 'HSF_Consumer'

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


def load_config(config_file):
    f = open(config_file,encoding='utf-8')
    res = f.read()
    config_content = json.loads(res)
    f.close()
    return config_content

def save_config(config,filename):
    config_content = {}
    for key,value in config.items():
        config_content[key] = value

    fw = open(filename,'w',encoding='utf-8')

    dic_json = json.dumps(config_content,ensure_ascii=False,indent=4)
    fw.write(dic_json)
    fw.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[Heter_Upgradation] AutoScaling')

    # , required=True
    parser.add_argument('--total_mode', type=int, default=0,
                        help='use total data of mean data')
    parser.add_argument('--qps_pred_col',type=str,default='y')
    parser.add_argument("--qps_err",type=int,default=0)

    # , required=True
    parser.add_argument("--qps_pred_dir",type=str,default='/data/k8smaster/CEA-Informer_atc/results')
    parser.add_argument("--qps_true_dir",type=str,default='exp_total_qps_data_aggr2/mean')
    parser.add_argument('--qps_pred_model',type=str,default='noeventrawinformer')
    parser.add_argument('--base_dir', type=str, default="hetergenous_data_exp/pod_metric_data_apps", help='data')
    parser.add_argument('--aggre_base_dir', type=str, default="hetergenous_data_exp/pod_metric_data_apps_aggr",
                        help='aggr_data')
    parser.add_argument("--heter_mode",type=int,default=0)
    parser.add_argument("--usage_mode",type=int,default=0)
    parser.add_argument('--pred_base_dir', type=str, default="hetergenous_data_exp/pod_metric_data_apps_aggr", help='data')

    parser.add_argument('--range_dir', type=str, default="./results_lsdd_no_update", help='data')
    parser.add_argument("--out_dir",type=str,default="./capacity_res",help="results")
    parser.add_argument("--base_model", type=str, default="826X")
    parser.add_argument("--start_date", type=str, default='20220815')
    parser.add_argument("--chunck_size", type=int, default=1)
    parser.add_argument("--freq", type=str, default='t')
    parser.add_argument("--step", type=int, default=1, help='the step when exploring the train data range')
    parser.add_argument("--diff_limit", type=float, default=0.15)
    parser.add_argument("--n_components", type=int, default=0, help="the limit of the n_components")
    parser.add_argument("--select_n", type=str, default='bic', help="the metric for the n_components decision")
    parser.add_argument("--diff_win", type=int, default=3, help="how to judge the convergence of GMM")
    parser.add_argument("--unit", type=str, default="ds", help="the col for the time")
    parser.add_argument("--test_size", type=float, default=0.2, help="the size of test data")
    parser.add_argument("--sample_time", type=str, default="sample_time_n")
    parser.add_argument("--norm", type=int, default=0)
    parser.add_argument("--x_unit", type=int, default=1)
    parser.add_argument('--util_unit',type=int,default=100)
    parser.add_argument("--y_unit", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=3)
    parser.add_argument("--standard", type=int, default=1)
    parser.add_argument("--gmm_mode", type=int, default=0)
    parser.add_argument("--norm_mode", type=int, default=2)
    parser.add_argument("--cov_col", type=str, default='cpu_util__pct')
    parser.add_argument("--pdc_col", type=str, default='RUE')
    parser.add_argument("--exp", type=int, default=100)
    parser.add_argument("--cap_exp",type=int,default=1)
    parser.add_argument("--idx",type=int,default=0)
    parser.add_argument('--iter',type=int,default=3)
    parser.add_argument('--reg',type=str,default='best')
    parser.add_argument("--perf_metric",type=str,default="mape")
    parser.add_argument("--end_date",type=str,default='20221020')
    parser.add_argument("--interval",type=int,default=6)
    parser.add_argument("--interval_unit",type=str,default="hour")
    parser.add_argument("--find_mode",type=int,default=0)
    parser.add_argument("--threshold_limit",type=float,default=0.15)
    parser.add_argument("--slid_step",type=int,default=8)
    parser.add_argument("--threshold_win",type=int,default=3)
    parser.add_argument("--additional",type=int,default=0)
    parser.add_argument("--least_period",type=int,default=1)
    parser.add_argument("--heter_update",type=int,default=0)
    # threshold_col='qps_usage_k'
    parser.add_argument("--threshold_col",type=str,default='rue')

    args = parser.parse_args()

    aim_apps = ['MS0']
   
    exp_range_dir = os.path.join(args.range_dir, str(args.exp))
    result_dir = os.path.join(args.out_dir,str(args.cap_exp))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir,exist_ok=True)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir,exist_ok=True)

    for ap in aim_apps[args.idx:(args.idx+11)]:
        app_path = os.path.join(args.base_dir, ap)
        aggre_app_path = os.path.join(args.aggre_base_dir, ap)
        pred_app_path = os.path.join(args.pred_base_dir,ap)
        range_dates_dir = os.path.join(os.path.join(args.range_dir,str(args.exp)),ap)
        app_range_res_path = os.path.join(exp_range_dir, ap)

        app_result_path = os.path.join(result_dir,ap)


        if not os.path.exists(app_result_path):
            os.makedirs(app_result_path, exist_ok=True)

        # aim_model_file = []
        aim_models = {}
        app_heter_data = {}
        aggre_app_heter_data = {}
        app_pred_heter_data = {}

        for j in os.listdir(app_path):
            if '.csv' in j and 'C3958' not in j:
                # aim_model_file.append(j)
                aim_models[j.split(".")[0]] = j
                app_heter_data[j.split(".")[0]] = pd.read_csv(os.path.join(app_path, j))
                aggre_app_heter_data[j.split(".")[0]] = pd.read_csv(os.path.join(aggre_app_path, j))
                print("get total train data from:")
                print(os.path.join(app_path, j))
                app_pred_heter_data[j.split(".")[0]] = pd.read_csv(os.path.join(pred_app_path, j))

                # container_app_name,throughput_type,ds,unit,
                # sample_time,minute,total_qps,total_usage,
                # cpu_util__pct,replicas,qps__Q_s,cpu_usage,rt__ms_Q,RUE,request_cpu

                if args.sample_time not in aggre_app_heter_data[j.split(".")[0]].columns:
                    aggre_app_heter_data[j.split(".")[0]] = aggre_app_heter_data[j.split(".")[0]].reset_index(drop=True)
                    aggre_app_heter_data[j.split(".")[0]][args.sample_time] = aggre_app_heter_data[j.split(".")[0]]["sample_time"].copy()

                aggre_app_heter_data[j.split(".")[0]] = aggre_app_heter_data[j.split(".")[0]][[
                    "throughput_type", "ds","minute", "unit","qps__Q_s","cpu_usage","cpu_util__pct","replicas",
                    "request_cpu",args.sample_time, 'RUE'
                ]]

                app_heter_data[j.split(".")[0]] = app_heter_data[j.split(".")[0]][[
                                             'throughput_type', 'ds', 'minute',
                                             'qps__Q_s', 'rt__ms_Q', 'cpu_usage', 'max_cpu_util',
                                             'replicas',
                                             'cpu_util__pct', 'request_cpu', args.sample_time, 'RUE']].reset_index(drop=True)
                app_heter_data[j.split(".")[0]].ds =  app_heter_data[j.split(".")[0]].ds.astype(int)
                app_heter_data[j.split(".")[0]].ds = app_heter_data[j.split(".")[0]].ds.astype(str)

                aggre_app_heter_data[j.split(".")[0]].ds = aggre_app_heter_data[j.split(".")[0]].ds.astype(int)
                aggre_app_heter_data[j.split(".")[0]].ds = aggre_app_heter_data[j.split(".")[0]].ds.astype(str)

                aggre_app_heter_data[j.split(".")[0]]['cpu_util__pct'] = aggre_app_heter_data[j.split(".")[0]]['cpu_util__pct']*args.util_unit


                app_heter_data[j.split(".")[0]][args.sample_time] = pd.to_datetime(
                    app_heter_data[j.split(".")[0]][args.sample_time])

                aggre_app_heter_data[j.split(".")[0]][args.sample_time] = pd.to_datetime(
                    aggre_app_heter_data[j.split(".")[0]][args.sample_time])

                # container_app_name,throughput_type,ds,unit,level_4,sample_time,minute,total_qps,
                # total_usage,cpu_util__pct,replicas,qps__Q_s,cpu_usage,rt__ms_Q,RUE,request_cpu

                app_pred_heter_data[j.split(".")[0]] = app_pred_heter_data[j.split(".")[0]][[
                    'throughput_type', 'ds', 'minute',"unit",
                    'qps__Q_s', 'rt__ms_Q', 'cpu_usage','total_qps','total_usage',
                    'replicas',
                    'cpu_util__pct', 'request_cpu', "sample_time", 'RUE']].dropna().reset_index(drop=True)

                app_pred_heter_data[j.split(".")[0]].ds = app_pred_heter_data[j.split(".")[0]].ds.astype(int)
                app_pred_heter_data[j.split(".")[0]].ds = app_pred_heter_data[j.split(".")[0]].ds.astype(str)
                app_pred_heter_data[j.split(".")[0]]['cpu_util__pct'] = app_pred_heter_data[j.split(".")[0]]['cpu_util__pct']*args.util_unit

                app_pred_heter_data[j.split(".")[0]].rename(columns={"sample_time":args.sample_time},inplace=True)

                app_pred_heter_data[j.split(".")[0]][args.sample_time] = pd.to_datetime(
                    app_pred_heter_data[j.split(".")[0]][args.sample_time])

        aim_dates = os.listdir(range_dates_dir)
        aim_dates.sort()
        # if args.qps_err > 0:
        true_data = pd.read_csv(os.path.join(args.qps_true_dir,"%s.csv" % ap))
        print("read total true data from:")
        print(os.path.join(args.qps_true_dir,"%s.csv" % ap))
        print(true_data.columns)
        true_data = true_data[['ds','sample_time',"unit",'num_containers','total_suage','total_qps','request']].dropna().reset_index(drop=True)

        true_data.rename(columns={"ds":"day","total_suage":"total_usage"},inplace=True)
        true_data["day"] = true_data["day"].astype(int)
        true_data["day"] = true_data["day"].astype(str)

        true_data["unit"] = true_data["unit"].astype(int)

        true_data.rename(columns={"sample_time":args.sample_time},inplace=True)
        # / float(args.util_unit)
        true_data['total_usage'] = true_data['total_usage']
        true_data['total_quota'] = true_data['request'] * true_data['num_containers']

        # print()
            # true_data.rename(co)
        test_qps_dir = os.path.join(os.path.join(args.qps_pred_dir,args.qps_pred_model),ap)
        setting_paths = os.listdir(test_qps_dir)
        setting_pio = {}
        for j in setting_paths:
            if os.path.isdir(os.path.join(test_qps_dir,j)):
                predict_time = int(j.split("_")[-1])
                setting_pio[predict_time] = j

        setting_times = list(setting_pio.keys())
        setting_times.sort()
        use_setting = setting_pio[setting_times[-1]]
        use_qps_pred_res_dir = os.path.join(test_qps_dir,use_setting)
        qps_pred_dates = os.listdir(use_qps_pred_res_dir)
        true_qps_dir = os.path.join(args.qps_true_dir,ap)

        print(app_heter_data.keys())
        print(aggre_app_heter_data.keys())

        for ad in aim_dates:
            if ad < args.start_date:
                continue
            if ad > args.end_date:
                continue
            if ad not in qps_pred_dates:
                continue
            else:
                app_date_result_path = os.path.join(app_result_path,ad)
                if not os.path.exists(app_date_result_path):
                    os.makedirs(app_date_result_path)
                data_range_conf_path = os.path.join(range_dates_dir,ad)
                pio_dates = {}
                for j in range(args.iter):
                    tmp_pio_data = pd.read_csv(os.path.join(data_range_conf_path,"aim_time_%s_%d.csv" % (args.base_model,j)))

                    tmp_pio_data = tmp_pio_data.sort_values("start",ascending=False).reset_index(drop=True)
                    # on_ds = []
                    last_start = tmp_pio_data.start.values[0]
                    last_ds = last_start.split()[0].split("-")
                    last_y = last_ds[0]
                    last_m = last_ds[1]
                    last_d = last_ds[2]
                    last_day = utils.which_day(year=int(last_y), month=int(last_m), date=int(last_d))
                    aim_differ = 0
                    aim_start = last_start
                    for da in tmp_pio_data.start.values:
                        da_ds = da.split()[0].split("-")
                        da_y = da_ds[0]
                        da_m = da_ds[1]
                        da_d = da_ds[2]
                        da_day = utils.which_day(year=int(da_y),month=int(da_m),date=int(da_d))
                        # on_ds.append(int(last_day - da_day))
                        now_differ = last_day - da_day
                        if now_differ > aim_differ:
                            break
                        else:
                            aim_start = da
                            aim_differ += 1

                    print("aim_start is:")
                    print(aim_start)

                    pio_dates[aim_start] = tmp_pio_data


                aim_pio_starts = list(pio_dates.keys())
                aim_pio_starts.sort()

                print("select start day from:")
                print(aim_pio_starts)

                selected_start_dates = pio_dates[aim_pio_starts[-1]]

                print("selected start day is:")
                print(selected_start_dates)

                selected_ds = utils.get_selected_ds(selected_start_dates,additional=int(args.additional))
                selected_ds.sort()
                print("selected train data ds range is:")
                print(selected_ds)

                selected_start_dates['start'] = pd.to_datetime(selected_start_dates['start'])
                selected_start_dates['end'] = pd.to_datetime(selected_start_dates['end'])

                using_data_dict = {}
                using_pred_data_dict = {}
                aggre_using_data_dict = {}
                other_models = []
                final_other_models = []

                true_qps_path = os.path.join(true_qps_dir, ad)

                test_data = pd.read_csv(os.path.join(os.path.join(use_qps_pred_res_dir, ad), "predict_results.csv"))

                qps_candidate = ["yhat","y_first","y_last","y_mean","y_first2","y_mean2","y_last2"]
                raw_qps_mape = 100.0
                # args.qps_pred_col = "yhat"
                qps_can_perf = {}
                for qc in qps_candidate:
                    if qc in test_data.columns:
                        tmp_mape = utils.MAPE(y_true=test_data["y"].to_numpy(),y_pred=test_data[qc].to_numpy())
                        qps_can_perf[tmp_mape] = qc
                qps_can_perf_values = list(qps_can_perf.keys())
                qps_can_perf_values.sort()
                args.qps_pred_col = qps_can_perf[qps_can_perf_values[0]]

                test_data = test_data[[args.qps_pred_col,"y","unit"]].dropna().reset_index(drop=True)

                test_data["hour"] = (test_data["unit"] // 12) + (len(selected_ds)*24)

                test_data["hour"] = test_data["hour"].astype(int)

                test_data['day'] = ad
                test_data = test_data.reset_index(drop=True)

                ad_true_data = true_data[true_data["day"] == ad].reset_index(drop=True)
                print("true_data shape is:")
                print(ad_true_data.shape)

                # ,args.sample_time
                # test_data.rename(columns={'ds': args.sample_time}, inplace=True)
                # test_data[args.sample_time] = pd.to_datetime(test_data[args.sample_time])
                ad_true_data[args.sample_time] = pd.to_datetime(ad_true_data[args.sample_time])
                print(test_data.columns)
                print(ad_true_data.columns)
                test_data["unit"] = test_data["unit"].astype(int)
                ad_true_data["unit"] = ad_true_data["unit"].astype(int)

                test_data["day"] = test_data["day"].astype(int)
                ad_true_data["day"] = ad_true_data["day"].astype(int)

                test_data["day"] = test_data["day"].astype(str)
                ad_true_data["day"] = ad_true_data["day"].astype(str)

                test_data = pd.merge(test_data, ad_true_data, on=['day',"unit"],how='inner').dropna()
                test_data = test_data.reset_index(drop=True)

                test_data.rename(columns={"day": "ds"}, inplace=True)

                if test_data.shape[0] < 1:
                    continue

                if args.qps_err > 0:
                    test_data[args.qps_pred_col] = test_data[args.qps_pred_col] / test_data['num_containers']
                    test_data["y"] = test_data["y"] / test_data['num_containers']

                print(app_heter_data.keys())
                print(aggre_app_heter_data.keys())
                for m in app_heter_data.keys():
                    selected_final = selected_ds+[ad]
                    selected_final.sort()
                    for kk in range(len(selected_final)):
                        tmp_data_s = app_heter_data[m][app_heter_data[m].ds.isin([selected_final[kk]])].reset_index(drop=True)
                        tmp_data_s['hour'] = tmp_data_s['minute'] // 60 + kk*24
                        tmp_data_s['hour'] = tmp_data_s['hour'].astype(int)

                        aggre_tmp_data_s = aggre_app_heter_data[m][aggre_app_heter_data[m].ds.isin([selected_final[kk]])].reset_index(
                            drop=True)
                        aggre_tmp_data_s['hour'] = aggre_tmp_data_s['unit'] // 12 + kk * 24
                        aggre_tmp_data_s['hour'] = aggre_tmp_data_s['hour'].astype(int)
                        if kk == 0:
                            using_data_dict[m] = tmp_data_s
                            aggre_using_data_dict[m] = aggre_tmp_data_s
                        else:
                            using_data_dict[m] = pd.concat([using_data_dict[m],tmp_data_s],axis=0).reset_index(drop=True)
                            aggre_using_data_dict[m] = pd.concat([aggre_using_data_dict[m],aggre_tmp_data_s],axis=0).reset_index(drop=True)

                    using_pred_data_dict[m] = app_pred_heter_data[m][app_pred_heter_data[m].ds.isin([ad])].reset_index(
                        drop=True)
                    using_pred_data_dict[m]['hour'] = using_pred_data_dict[m]['unit'] // 12 + len(selected_ds)*24
                    using_pred_data_dict[m]['hour'] = using_pred_data_dict[m]['hour'].astype(int)

                    # using_data_dict[m] = app_heter_data[m][app_heter_data[m].ds.isin(selected_ds+[ad])].reset_index(drop=True)

                    # aggre_using_data_dict[m] = aggre_app_heter_data[m][aggre_app_heter_data[m].ds.isin(selected_ds+[ad])].reset_index(drop=True)
                    # using_pred_data_dict[m] = app_pred_heter_data[m][app_pred_heter_data[m].ds.isin([ad])].reset_index(drop=True)
                    if using_data_dict[m].shape[0] < 1 or using_pred_data_dict[m].shape[0] < 1 or aggre_using_data_dict[m].shape[0] < 1:
                        using_data_dict.pop(m)
                        aggre_using_data_dict.pop(m)
                        using_pred_data_dict.pop(m)
                    else:
                        using_pred_data_dict[m].rename(columns={'total_qps':"%s_total_qps" % m,
                                                                'total_usage':"%s_total_usage" % m,
                                                                'replicas':'%s_replicas' % m,
                                                                "request_cpu":"%s_request" % m
                                                                },inplace=True)

                        if m != args.base_model:
                            other_models.append(m)

                using_data_dict[args.base_model]['base_cpu_util'] = using_data_dict[args.base_model]['cpu_util__pct'].copy()
                using_data_dict[args.base_model]['base_RUE'] = using_data_dict[args.base_model]['RUE'].copy()

                aggre_using_data_dict[args.base_model]['base_cpu_util'] = aggre_using_data_dict[args.base_model][
                    'cpu_util__pct'].copy()
                aggre_using_data_dict[args.base_model]['base_RUE'] = aggre_using_data_dict[args.base_model]['RUE'].copy()


                print(other_models)


                result_config = {}
                result_config['qps_type'] = args.qps_pred_col

                final_stat_hour = len(selected_ds)*24
                t_step = int(24//args.interval)

                for m in other_models:
                    using_data_dict[m] = pd.merge(using_data_dict[m],using_data_dict[args.base_model][[
                        args.sample_time,"base_cpu_util","base_RUE"
                    ]],on=args.sample_time).dropna().reset_index(drop=True)

                    using_data_dict[m]['map'] = using_data_dict[m]['base_RUE'] / using_data_dict[m]['RUE']

                    aggre_using_data_dict[m] = pd.merge(aggre_using_data_dict[m], aggre_using_data_dict[args.base_model][[
                        args.sample_time, "base_cpu_util", "base_RUE"
                    ]], on=args.sample_time).dropna().reset_index(drop=True)

                    aggre_using_data_dict[m]['map'] = aggre_using_data_dict[m]['base_RUE'] / aggre_using_data_dict[m]['RUE']

                perf_maps = {}

                total_out_data = pd.DataFrame()

                result_config['total'] = {}

                result_config['total']["heter"] = {}
                result_config['total']['usage'] = {}
                result_config['total']["qps"] = {}

                for kk in range(0,24,t_step):
                    now_s = int(kk)
                    result_config[now_s] = {}
                    perf_maps[now_s] = {}
                    tmp_using_data_dict = {}
                    tmp_aggre_using_data_dict = {}
                    tmp_using_pred_data_dict = {}

                    tmp_other_models = other_models[:]

                    for m in using_data_dict.keys():
                        tmp_using_data_dict[m] = using_data_dict[m][
                            using_data_dict[m].hour<(final_stat_hour+kk+t_step)].reset_index(drop=True)
                        tmp_aggre_using_data_dict[m] = aggre_using_data_dict[m][
                            aggre_using_data_dict[m].hour<(final_stat_hour+kk+t_step)].reset_index(drop=True)
                        tmp_using_pred_data_dict[m] = using_pred_data_dict[m][
                            (using_pred_data_dict[m].hour>=(final_stat_hour+kk))
                            &(using_pred_data_dict[m].hour<(final_stat_hour+kk+t_step))].reset_index(drop=True)

                        if tmp_using_data_dict[m].shape[0] < 1:
                            if m in tmp_using_data_dict.keys():
                                tmp_using_data_dict.pop(m)
                            if m in tmp_aggre_using_data_dict.keys():
                                tmp_aggre_using_data_dict.pop(m)
                            if m in tmp_using_pred_data_dict.keys():
                                tmp_using_pred_data_dict.pop(m)

                            if m in tmp_other_models:
                                tmp_other_models.remove(m)


                            # if m == args.base_model:
                            #     continue
                        if m in tmp_aggre_using_data_dict.keys():
                            if (tmp_aggre_using_data_dict[m][(tmp_aggre_using_data_dict[m].hour >= (final_stat_hour + kk))
                                                         & (tmp_aggre_using_data_dict[m].hour < (
                                        final_stat_hour + kk + t_step))].shape[0] < 1) or (
                                    tmp_aggre_using_data_dict[m][
                                        (tmp_aggre_using_data_dict[m].hour < (final_stat_hour + kk))
                                        ].shape[0] < 1
                            ):

                                if m in tmp_using_data_dict.keys():
                                    tmp_using_data_dict.pop(m)
                                if m in tmp_aggre_using_data_dict.keys():
                                    tmp_aggre_using_data_dict.pop(m)
                                if m in tmp_using_pred_data_dict.keys():
                                    tmp_using_pred_data_dict.pop(m)

                                if m in tmp_other_models:
                                    tmp_other_models.remove(m)

                        if m in tmp_using_data_dict.keys():
                            if (tmp_using_data_dict[m][(tmp_using_data_dict[m].hour >= (final_stat_hour + kk))
                                                         & (tmp_using_data_dict[m].hour < (
                                        final_stat_hour + kk + t_step))].shape[0] < 1) or (
                                    tmp_using_data_dict[m][
                                        (tmp_using_data_dict[m].hour < (final_stat_hour + kk))
                                        ].shape[0] < 1
                            ):

                                if m in tmp_using_data_dict.keys():
                                    tmp_using_data_dict.pop(m)
                                if m in tmp_aggre_using_data_dict.keys():
                                    tmp_aggre_using_data_dict.pop(m)
                                if m in tmp_using_pred_data_dict.keys():
                                    tmp_using_pred_data_dict.pop(m)

                                if m in tmp_other_models:
                                    tmp_other_models.remove(m)

                        if m in tmp_using_pred_data_dict.keys():
                            if tmp_using_pred_data_dict[m].shape[0] < 1:
                                if m in tmp_using_data_dict.keys():
                                    tmp_using_data_dict.pop(m)
                                if m in tmp_aggre_using_data_dict.keys():
                                    tmp_aggre_using_data_dict.pop(m)
                                if m in tmp_using_pred_data_dict.keys():
                                    tmp_using_pred_data_dict.pop(m)

                                if m in tmp_other_models:
                                    tmp_other_models.remove(m)


                    if (args.base_model not in tmp_using_pred_data_dict.keys()) or (
                            args.base_model not in tmp_aggre_using_data_dict.keys()) or (
                            args.base_model not in tmp_using_pred_data_dict.keys()):
                        continue
                    else:
                        if (tmp_using_data_dict[args.base_model].shape[0] < 1) or (
                                tmp_aggre_using_data_dict[args.base_model].shape[0] < 1
                        ) or (tmp_using_pred_data_dict[args.base_model].shape[0] < 1):
                            continue

                    result_config[now_s]["heter"] = {}

                    pred_datas = pd.merge(tmp_using_pred_data_dict[args.base_model][
                                    [args.sample_time,
                                    "unit","hour","%s_total_qps" % args.base_model,
                                    "%s_total_usage" % args.base_model,
                                    "%s_replicas" % args.base_model]]
                                      ,test_data[["unit","total_qps",'total_usage',
                                    "num_containers",args.qps_pred_col,"y"]],on=['unit'],how='inner').reset_index(drop=True)

                    pred_datas['%s_total_qps' % args.base_model] = pred_datas['%s_total_qps' % args.base_model].fillna(0)
                    pred_datas['%s_total_usage' % args.base_model] = pred_datas['%s_total_usage' % args.base_model].fillna(0)
                    pred_datas['%s_replicas' % args.base_model] = pred_datas['%s_replicas' % args.base_model].fillna(0)

                    pred_datas['real_qps' ] = pred_datas['%s_total_qps' % args.base_model].copy()
                    pred_datas['real_usage' ] = pred_datas['%s_total_usage' % args.base_model].copy()
                    pred_datas['real_replicas'] = pred_datas['%s_replicas' % args.base_model].copy()

                    max_mapes = 0.0
                    for m in tmp_other_models:
                        pred_datas = pd.merge(pred_datas, tmp_using_pred_data_dict[m][[
                        'unit', '%s_total_qps' % m, '%s_total_usage' % m, '%s_replicas' % m
                        ]], on='unit', how="left").reset_index(drop=True)

                        pred_datas['%s_total_qps' % m] = pred_datas['%s_total_qps' % m].fillna(0)
                        pred_datas['%s_total_usage' % m] = pred_datas['%s_total_usage' % m].fillna(0)
                        pred_datas['%s_replicas' % m] = pred_datas['%s_replicas' % m].fillna(0)

                        pred_datas['real_qps'] = pred_datas['real_qps'] + pred_datas['%s_total_qps' % m]
                        pred_datas['real_usage'] = pred_datas['real_usage'] + pred_datas['%s_total_usage' % m]
                        pred_datas['real_replicas'] = pred_datas['real_replicas'] + pred_datas['%s_replicas' % m]

                        perf_maps[now_s][m] = UsageMapPredictor()
                        if args.heter_mode > 1:
                            if args.heter_mode > 2:
                                map_x_train, map_x_test, map_y_train, map_y_test, mean_stds = utils.preprocess_qps_usage_data(
                                    aggre_using_data_dict[m].reset_index(drop=True),
                                    test_size=args.test_size, x_column="base_cpu_util", y_column="map",
                                    norm=False)
                                print("aggre heter training:")
                                print(aggre_using_data_dict[m].shape)
                                print(aggre_using_data_dict[m][aggre_using_data_dict[m].ds.isin([ad])].shape)
                                print(aggre_using_data_dict[m][args.sample_time].max())
                                print(aggre_using_data_dict[m][args.sample_time].min())
                            else:
                                map_x_train, map_x_test, map_y_train, map_y_test, mean_stds = utils.preprocess_qps_usage_data(
                                    tmp_aggre_using_data_dict[m][(tmp_aggre_using_data_dict[m].hour<(final_stat_hour+kk))
                                    ].reset_index(drop=True),
                                    test_size=args.test_size, x_column="base_cpu_util", y_column="map",
                                    norm=False)
                                print("aggre heter training:")
                                print(tmp_aggre_using_data_dict[m][
                                          (tmp_aggre_using_data_dict[m].hour < (final_stat_hour + kk))
                                      ][args.sample_time].shape)
                                print(tmp_aggre_using_data_dict[m][(tmp_aggre_using_data_dict[m].hour<(final_stat_hour+kk))
                                    ][args.sample_time].max())
                                print(tmp_aggre_using_data_dict[m][
                                          (tmp_aggre_using_data_dict[m].hour < (final_stat_hour + kk))
                                      ][args.sample_time].min())
                        else:
                            if args.heter_mode > 0:
                                map_x_train, map_x_test, map_y_train, map_y_test, mean_stds = utils.preprocess_qps_usage_data(
                                    using_data_dict[m],
                                    test_size=args.test_size, x_column="base_cpu_util", y_column="map",
                                    norm=False)
                                print("raw heter training:")
                                print(using_data_dict[m].shape)
                                print(using_data_dict[m][using_data_dict[m].ds.isin([ad])].shape)
                                print(using_data_dict[m][args.sample_time].max())
                                print(using_data_dict[m][args.sample_time].min())
                            else:
                                map_x_train, map_x_test, map_y_train, map_y_test, mean_stds = utils.preprocess_qps_usage_data(
                                    tmp_using_data_dict[m][(tmp_using_data_dict[m].hour<(final_stat_hour+kk))].reset_index(drop=True),
                                    test_size=args.test_size, x_column="base_cpu_util", y_column="map",
                                    norm=False)
                                print("raw heter training:")
                                print(tmp_using_data_dict[m][(tmp_using_data_dict[m].hour < (final_stat_hour + kk))][
                                          args.sample_time].shape)
                                print(tmp_using_data_dict[m][(tmp_using_data_dict[m].hour<(final_stat_hour+kk))][args.sample_time].max())
                                print(tmp_using_data_dict[m][(tmp_using_data_dict[m].hour < (final_stat_hour + kk))][
                                          args.sample_time].min())

                        perf_maps[now_s][m].train(X_train=map_x_train,
                                       y_train=map_y_train)
                        # os.path.join(app_date_result_path,"result.json")
                        # "dpg_%d_%s.m"
                        joblib.dump(perf_maps[now_s][m].ll_forest, os.path.join(app_date_result_path,
                                                                      "dpg_ll_%d_%s.m" % (now_s,m)))
                        # joblib.dump(perf_maps[now_s][m].qu_forest, os.path.join(app_date_result_path,
                        #                                                         "dpg_qu_%d_%s.m" % (now_s, m)))

                        test_maps = perf_maps[now_s][m].predict_ll(x_test=map_x_test)

                        raw_maps0 = tmp_aggre_using_data_dict[m][(tmp_aggre_using_data_dict[m].hour>=(final_stat_hour+kk))
                            &(tmp_aggre_using_data_dict[m].hour<(final_stat_hour+kk+t_step))]["map"]
                        raw_cpu0 = tmp_aggre_using_data_dict[m][(tmp_aggre_using_data_dict[m].hour>=(final_stat_hour+kk))
                            &(tmp_aggre_using_data_dict[m].hour<(final_stat_hour+kk+t_step))][["base_cpu_util"]]
                        print("raw_maps0 shape:")
                        print(raw_maps0.shape)
                        print("raw_cpu0 shape:")
                        print(raw_cpu0.shape)
                        raw_pred_map0 = perf_maps[now_s][m].predict_ll(raw_cpu0)
                        result_config[now_s]['heter'][m] = {}
                        result_config[now_s]['heter'][m]['mape_raw'] = utils.MAPE(raw_maps0, raw_pred_map0)
                        result_config[now_s]['heter'][m]['mse_raw'] = MSE(raw_maps0, raw_pred_map0)
                        result_config[now_s]['heter'][m]['mae_raw'] = MAE(raw_maps0, raw_pred_map0)

                        tmp_aggre_using_data_dict[m] = tmp_aggre_using_data_dict[m][
                            tmp_aggre_using_data_dict[m].hour < (final_stat_hour+kk)].reset_index(
                            drop=True)
                        tmp_using_data_dict[m] = tmp_using_data_dict[m][
                            tmp_using_data_dict[m].hour < (final_stat_hour+kk)].reset_index(drop=True)

                        if args.usage_mode > 0:
                            maps = perf_maps[now_s][m].predict_ll(
                                tmp_aggre_using_data_dict[m][["base_cpu_util"]].to_numpy())
                            tmp_aggre_using_data_dict["pred_map"] = pd.Series(maps)

                            tmp_mape0 = utils.MAPE(y_true=map_y_test, y_pred=test_maps)
                            if tmp_mape0 > max_mapes:
                                max_mapes = tmp_mape0

                            tmp_aggre_using_data_dict[m]["y_usage"] = tmp_aggre_using_data_dict[m]['cpu_usage'] * \
                                                              tmp_aggre_using_data_dict[m]["pred_map"]
                            print("other train input data shape:")
                            print(tmp_aggre_using_data_dict[m].shape)
                            print("other train input data columns:")
                            print(tmp_aggre_using_data_dict[m].columns)

                            print("raw other train input data shape:")
                            print(aggre_using_data_dict[m].shape)
                            print("raw other train input data columns:")
                            print(aggre_using_data_dict[m].columns)
                        else:
                            maps = perf_maps[now_s][m].predict_ll(
                                tmp_using_data_dict[m][["base_cpu_util"]].to_numpy())
                            tmp_using_data_dict[m]["pred_map"] = pd.Series(maps)

                            tmp_mape0 = utils.MAPE(y_true=map_y_test, y_pred=test_maps)
                            if tmp_mape0 > max_mapes:
                                max_mapes = tmp_mape0

                            tmp_using_data_dict[m]["y_usage"] = tmp_using_data_dict[m]['cpu_usage'] * \
                                                                tmp_using_data_dict[m]["pred_map"]
                            print("other train input data shape:")
                            print(tmp_using_data_dict[m].shape)
                            print("other train input data columns:")
                            print(tmp_using_data_dict[m].columns)

                            print("raw other train input data shape:")
                            print(using_data_dict[m].shape)
                            print("raw other train input data columns:")
                            print(using_data_dict[m].columns)

                    if args.usage_mode > 0:
                        tmp_aggre_using_data_dict[args.base_model] = tmp_aggre_using_data_dict[args.base_model][
                            tmp_aggre_using_data_dict[args.base_model].hour < (final_stat_hour+kk)].reset_index(drop=True)
                        print("train input data shape:")
                        print(tmp_aggre_using_data_dict[args.base_model].shape)
                        print("train input data columns:")
                        print(tmp_aggre_using_data_dict[args.base_model].columns)

                        tmp_aggre_using_data_dict[args.base_model]['y_usage'] = tmp_aggre_using_data_dict[
                            args.base_model]['cpu_usage'].copy()

                    else:
                        tmp_using_data_dict[args.base_model] = tmp_using_data_dict[args.base_model][
                            tmp_using_data_dict[args.base_model].hour < (final_stat_hour+kk)].reset_index(drop=True)
                        print("train input data shape:")
                        print(tmp_using_data_dict[args.base_model].shape)
                        print("train input data columns:")
                        print(tmp_using_data_dict[args.base_model].columns)

                        tmp_using_data_dict[args.base_model]['y_usage'] = tmp_using_data_dict[args.base_model]['cpu_usage'].copy()

                    total_usage_lists = []

                    if args.heter_update < 1:
                        heter_update = False
                    else:
                        heter_update = True

                    # find_range_pdf = {}
                    if args.usage_mode > 0:
                        if not heter_update:
                            train_data_range = find_train_range_threshold(x_df_dict=tmp_aggre_using_data_dict,
                                                                          base_model=args.base_model,
                                                                          init_size=args.least_period * 24, step=args.slid_step,
                                                                          threshold_limit=args.threshold_limit,
                                                                          threshold_win=args.threshold_win,
                                                                          unit=args.interval_unit, x_col='qps__Q_s',
                                                                          y_col='y_usage', test_size=0.1,
                                                                          sample_time=args.sample_time, norm=False,
                                                                          x_unit=1, y_unit=1000,
                                                                          block_size=args.least_period * 24,
                                                                          cov_col="cpu_util__pct", pdc_col='RUE',
                                                                          threshold_col=args.threshold_col,
                                                                          find_mode=args.find_mode,
                                                                          heter_update=heter_update
                                                                          )
                        else:
                            train_data_range = find_train_range_threshold(x_df_dict=tmp_aggre_using_data_dict,
                                                                          base_model=args.base_model,
                                                                          init_size=args.least_period * 24, step=args.slid_step,
                                                                          threshold_limit=args.threshold_limit,
                                                                          threshold_win=args.threshold_win,
                                                                          unit=args.interval_unit, x_col='qps__Q_s',
                                                                          y_col='cpu_usage', test_size=0.1,
                                                                          sample_time=args.sample_time, norm=False,
                                                                          x_unit=1, y_unit=1000,
                                                                          block_size=args.least_period * 24,
                                                                          cov_col="cpu_util__pct", pdc_col='RUE',
                                                                          threshold_col=args.threshold_col,
                                                                          find_mode=args.find_mode,
                                                                          heter_update=heter_update
                                                                          )

                        train_data_range.sort()
                        print(len(train_data_range))
                        print(len(tmp_aggre_using_data_dict[args.base_model].hour.unique().tolist()))
                        print(train_data_range[0])
                        print(train_data_range[-1])

                        for j in list(tmp_aggre_using_data_dict.keys()):
                            total_usage_lists.append(tmp_aggre_using_data_dict[j][
                                                         tmp_aggre_using_data_dict[j][args.interval_unit].isin(train_data_range)
                                                     ][[args.sample_time,'qps__Q_s', 'y_usage']].reset_index(drop=True))
                            pred_datas = pred_datas.reset_index(drop=True)
                            pred_datas["%s_input" % j] = pred_datas[args.qps_pred_col] * pred_datas["%s_total_qps" % j] / (
                                    pred_datas['real_qps'] + 1e-3)
                            pred_datas["%s_true_qps" % j] = pred_datas["y"] * pred_datas[
                                "%s_total_qps" % j] / (
                                                                    pred_datas['real_qps'] + 1e-3)

                            pred_datas["%s_ground_usage" % j] = pred_datas["total_usage"] * pred_datas[
                            "%s_total_usage" % j] / (pred_datas['real_usage'] + 1e-4)
                            pred_datas["%s_ground_qps" % j] = pred_datas["total_qps"] * pred_datas["%s_total_qps" % j] / (
                                    pred_datas['real_qps'] + 1e-3)

                            pred_datas = pred_datas.reset_index(drop=True)
                            print(tmp_using_pred_data_dict[j].head())
                            print(tmp_using_pred_data_dict[j].shape)
                            print(tmp_using_pred_data_dict[j].dropna().shape)
                            tmp_using_pred_data_dict[j] = pd.merge(tmp_using_pred_data_dict[j].reset_index(drop=True),
                                                           pred_datas[["unit", "%s_input" % j, "real_replicas",
                                                                       "num_containers",
                                                                       "%s_ground_usage" % j,
                                                                       "%s_ground_qps" % j,"%s_true_qps" % j]], on='unit', how='inner')

                            print("pred base data is:")
                            print(pred_datas[["unit", "%s_input" % j, "real_replicas",
                                          "num_containers",
                                          "%s_ground_usage" % j,
                                          "%s_ground_qps" % j,"%s_true_qps" % j]].head())
                            print(pred_datas[["unit", "%s_input" % j, "real_replicas",
                                          "num_containers",
                                          "%s_ground_usage" % j,
                                          "%s_ground_qps" % j,"%s_true_qps" % j]].shape)
                            print(pred_datas[["unit", "%s_input" % j, "real_replicas",
                                          "num_containers",
                                          "%s_ground_usage" % j,
                                          "%s_ground_qps" % j,"%s_true_qps" % j]].dropna().shape)

                            tmp_using_pred_data_dict[j]["%s_pred_replicas" % j] = tmp_using_pred_data_dict[j]["num_containers"] / \
                                                                          tmp_using_pred_data_dict[j]["real_replicas"] * \
                                                                          tmp_using_pred_data_dict[j]["%s_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_input" % j] = tmp_using_pred_data_dict[j]["%s_input" % j] / \
                                                                  tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_true_qps" % j] = tmp_using_pred_data_dict[j]["%s_true_qps" % j] / \
                                                                          tmp_using_pred_data_dict[j][
                                                                              "%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_ground_qps" % j] = tmp_using_pred_data_dict[j]["%s_ground_qps" % j] / \
                                                                       tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_ground_usage" % j] = tmp_using_pred_data_dict[j][
                                                                             "%s_ground_usage" % j] / \
                                                                         tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_total_ground_usage" % j] = tmp_using_pred_data_dict[j][
                                                                                   "%s_ground_usage" % j] * \
                                                                               tmp_using_pred_data_dict[j][
                                                                                   "%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_total_ground_qps" % j] = tmp_using_pred_data_dict[j][
                                                                                 "%s_ground_qps" % j] * \
                                                                             tmp_using_pred_data_dict[j][
                                                                                 "%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_pred_quota" % j] = tmp_using_pred_data_dict[j]["%s_request" % j] * \
                                                                       tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            print(tmp_using_pred_data_dict[j].shape)
                            print(tmp_using_pred_data_dict[j].dropna().shape)
                            print("%s input data shape is:" % j)
                            print(tmp_using_pred_data_dict[j].shape)
                    else:
                        if not heter_update:
                            train_data_range = find_train_range_threshold(x_df_dict=tmp_using_data_dict,
                                                                          base_model=args.base_model,
                                                                          init_size=args.least_period * 24, step=args.slid_step,
                                                                          threshold_limit=args.threshold_limit,
                                                                          threshold_win=args.threshold_win,
                                                                          unit=args.interval_unit, x_col='qps__Q_s',
                                                                          y_col='y_usage', test_size=0.1,
                                                                          sample_time=args.sample_time, norm=False,
                                                                          x_unit=1, y_unit=1000,
                                                                          block_size=args.least_period * 24,
                                                                          cov_col="cpu_util__pct", pdc_col='RUE',
                                                                          threshold_col=args.threshold_col,
                                                                          find_mode=args.find_mode,
                                                                          heter_update=heter_update
                                                                          )
                        else:
                            train_data_range = find_train_range_threshold(x_df_dict=tmp_using_data_dict,
                                                                          base_model=args.base_model,
                                                                          init_size=args.least_period * 24, step=args.slid_step,
                                                                          threshold_limit=args.threshold_limit,
                                                                          threshold_win=args.threshold_win,
                                                                          unit=args.interval_unit, x_col='qps__Q_s',
                                                                          y_col='cpu_usage', test_size=0.1,
                                                                          sample_time=args.sample_time, norm=False,
                                                                          x_unit=1, y_unit=1000,
                                                                          block_size=args.least_period * 24,
                                                                          cov_col="cpu_util__pct", pdc_col='RUE',
                                                                          threshold_col=args.threshold_col,
                                                                          find_mode=args.find_mode,
                                                                          heter_update=heter_update
                                                                          )

                        train_data_range.sort()
                        print(len(train_data_range))
                        print(len(tmp_using_data_dict[args.base_model].hour.unique().tolist()))
                        print(train_data_range[0])
                        print(train_data_range[-1])
                        for j in list(tmp_using_data_dict.keys()):
                            total_usage_lists.append(tmp_using_data_dict[j][
                                                         tmp_using_data_dict[j][args.interval_unit].isin(train_data_range)
                                                     ][[args.sample_time,'qps__Q_s', 'y_usage']].reset_index(drop=True))
                            pred_datas = pred_datas.reset_index(drop=True)
                            pred_datas["%s_input" % j] = pred_datas[args.qps_pred_col] * pred_datas["%s_total_qps" % j] / (
                                    pred_datas['real_qps'] + 1e-3)
                            pred_datas["%s_true_qps" % j] = pred_datas["y"] * pred_datas[
                                "%s_total_qps" % j] / (pred_datas['real_qps'] + 1e-3)
                            pred_datas["%s_ground_usage" % j] = pred_datas["total_usage"] * pred_datas[
                                "%s_total_usage" % j] / (pred_datas['real_usage'] + 1e-4)
                            pred_datas["%s_ground_qps" % j] = pred_datas["total_qps"] * pred_datas["%s_total_qps" % j] / (
                                    pred_datas['real_qps'] + 1e-3)

                            pred_datas = pred_datas.reset_index(drop=True)

                            print(tmp_using_pred_data_dict[j].head())
                            print(tmp_using_pred_data_dict[j].shape)
                            print(tmp_using_pred_data_dict[j].dropna().shape)

                            tmp_using_pred_data_dict[j] = pd.merge(tmp_using_pred_data_dict[j].reset_index(drop=True),
                                                           pred_datas[["unit", "%s_input" % j, "real_replicas",
                                                                       "num_containers",
                                                                       "%s_ground_usage" % j,
                                                                       "%s_ground_qps" % j,"%s_true_qps" % j]], on='unit', how='inner')

                            print("pred base data is:")
                            print(pred_datas[["unit", "%s_input" % j, "real_replicas",
                                          "num_containers",
                                          "%s_ground_usage" % j,
                                          "%s_ground_qps" % j,"%s_true_qps" % j]].head())

                            print(pred_datas[["unit", "%s_input" % j, "real_replicas",
                                          "num_containers",
                                          "%s_ground_usage" % j,
                                          "%s_ground_qps" % j,"%s_true_qps" % j]].shape)

                            print(pred_datas[["unit", "%s_input" % j, "real_replicas",
                                          "num_containers",
                                          "%s_ground_usage" % j,
                                          "%s_ground_qps" % j,"%s_true_qps" % j]].dropna().shape)

                            tmp_using_pred_data_dict[j]["%s_pred_replicas" % j] = tmp_using_pred_data_dict[j]["num_containers"] / \
                                                                          tmp_using_pred_data_dict[j]["real_replicas"] * \
                                                                          tmp_using_pred_data_dict[j]["%s_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_input" % j] = tmp_using_pred_data_dict[j]["%s_input" % j] / \
                                                                  tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_true_qps" % j] = tmp_using_pred_data_dict[j]["%s_true_qps" % j] / \
                                                                          tmp_using_pred_data_dict[j][
                                                                              "%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_ground_qps" % j] = tmp_using_pred_data_dict[j]["%s_ground_qps" % j] / \
                                                                       tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_ground_usage" % j] = tmp_using_pred_data_dict[j][
                                                                             "%s_ground_usage" % j] / \
                                                                         tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_total_ground_usage" % j] = tmp_using_pred_data_dict[j][
                                                                                   "%s_ground_usage" % j] * \
                                                                               tmp_using_pred_data_dict[j][
                                                                                   "%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_total_ground_qps" % j] = tmp_using_pred_data_dict[j][
                                                                                 "%s_ground_qps" % j] * \
                                                                             tmp_using_pred_data_dict[j][
                                                                                 "%s_pred_replicas" % j]

                            tmp_using_pred_data_dict[j]["%s_pred_quota" % j] = tmp_using_pred_data_dict[j]["%s_request" % j] * \
                                                                       tmp_using_pred_data_dict[j]["%s_pred_replicas" % j]

                            print(tmp_using_pred_data_dict[j].shape)
                            print(tmp_using_pred_data_dict[j].dropna().shape)
                            print("%s input data shape is:" % j)
                            print(tmp_using_pred_data_dict[j].shape)

                    print("hetergenous test input data shape:")
                    print(tmp_using_pred_data_dict[args.base_model].shape)
                    print("hetergenous test input data columns:")
                    print(tmp_using_pred_data_dict[args.base_model].columns)

                    print("heter mape is: %f" % max_mapes)

                    if max_mapes >= 0.2:
                        if args.usage_mode > 0:
                            total_usage_df = tmp_aggre_using_data_dict[args.base_model][
                                tmp_aggre_using_data_dict[args.base_model][args.interval_unit].isin(train_data_range)][[args.sample_time,'qps__Q_s', 'y_usage']].reset_index(
                                drop=True)
                        else:
                            total_usage_df = tmp_using_data_dict[args.base_model][
                                tmp_using_data_dict[args.base_model][args.interval_unit].isin(train_data_range)][[args.sample_time,'qps__Q_s', 'y_usage']].reset_index(drop=True)
                    else:
                        total_usage_df = pd.concat(total_usage_lists,axis=0).reset_index(drop=True)

                    print("total dataset for capacity model with shape:")
                    print(total_usage_df.shape)
                    print(total_usage_df.dropna().shape)

                    total_usage_df = total_usage_df.dropna().reset_index(drop=True)

                    result_config[now_s]['start'] = str(total_usage_df[args.sample_time].min())
                    result_config[now_s]['start_raw'] = str(tmp_using_data_dict[args.base_model][args.sample_time].min())

                    print("hour: %d true start: %s" % (now_s,result_config[now_s]['start']))
                    print("hour: %d raw start: %s" % (now_s, result_config[now_s]['start_raw']))

                    total_usage_df = total_usage_df[['qps__Q_s', 'y_usage']].reset_index(drop=True)

                    # result_config[now_s]['start_ds'] = tmp_using_data_dict[args.base_model].sort_values(args.sample_time).head()

                    norm = False
                    if args.norm >= 1:
                        norm = True
                    else:
                        norm = False

                    usage_predictor = PredictUsageModel(reg_x_metric='qps__Q_s',reg_y_metric='y_usage',
                                                        perf_metric=args.perf_metric,norm=norm,test_size=args.test_size)

                    if 'best' not in args.reg:
                        usage_predictor.train(total_usage_df,stage='regressor',test_size=args.test_size,reg=args.reg)
                    else:
                        usage_predictor.train(total_usage_df, stage='regressor', test_size=args.test_size, reg=None)

                    print(tmp_using_pred_data_dict[args.base_model].head())
                    print(tmp_using_pred_data_dict[args.base_model].shape)
                    print(tmp_using_pred_data_dict[args.base_model].dropna().shape)
                    save_config(usage_predictor.reg_performance,
                                os.path.join(app_date_result_path, "reg_result_%d.json" % now_s))
                    save_config(usage_predictor.mean_stds,
                                os.path.join(app_date_result_path, "mean_std_result_%d.json" % now_s))

                    result_config[now_s]['reg'] = usage_predictor.reg_models["best"]

                    base_results = usage_predictor.predict_usage(future_qps=tmp_using_pred_data_dict[args.base_model][[
                    "%s_input" % args.base_model]
                    ], reg=args.reg)

                    # tmp_using_pred_data_dict[args.base_model]['%s_pred_usage' % args.base_model] = pd.Series(
                    #     base_results)

                    tmp_using_pred_data_dict[args.base_model] = tmp_using_pred_data_dict[args.base_model].reset_index(
                        drop=True)

                    tmp_using_pred_data_dict[args.base_model]['%s_pred_usage' % args.base_model] = pd.Series(
                        base_results)

                    usage_pred_regs = list(usage_predictor.reg_performance.keys())

                    if args.reg not in usage_pred_regs:
                        usage_pred_regs.append(args.reg)

                    for regg in usage_pred_regs:
                        if (regg in usage_predictor.reg_models.keys()) and ("best" not in regg) and ('rfr' not in regg):
                            joblib.dump(usage_predictor.reg_models[regg], os.path.join(app_date_result_path,
                                                                                "reg_%s_%d.m" % (regg,now_s)))

                        base_true_results = usage_predictor.predict_usage(
                            future_qps=tmp_using_pred_data_dict[args.base_model][[
                                "%s_true_qps" % (args.base_model)]
                            ], reg=regg)

                        tmp_using_pred_data_dict[args.base_model][
                            '%s_true_qps_usage_%s' % (args.base_model, regg)] = pd.Series(
                            base_true_results)

                        base_reg_results = usage_predictor.predict_usage(
                            future_qps=tmp_using_pred_data_dict[args.base_model][[
                                "%s_input" % args.base_model]
                            ], reg=regg)

                        tmp_using_pred_data_dict[args.base_model][
                            '%s_pred_usage_%s' % (args.base_model, regg)] = pd.Series(
                            base_reg_results)

                        tmp_using_pred_data_dict[args.base_model]['%s_total_true_qps_usage_%s'
                                                                  % (args.base_model, regg)] = \
                            tmp_using_pred_data_dict[args.base_model]['%s_true_qps_usage_%s' % (args.base_model, regg)] * \
                            tmp_using_pred_data_dict[args.base_model]['%s_pred_replicas'
                                                                      % args.base_model]

                        tmp_using_pred_data_dict[args.base_model]['%s_total_pred_usage_%s'
                                                                  % (args.base_model, regg)] = \
                            tmp_using_pred_data_dict[args.base_model][
                                '%s_pred_usage_%s' % (args.base_model, regg)] * \
                            tmp_using_pred_data_dict[args.base_model]['%s_pred_replicas'
                                                                      % args.base_model]

                    tmp_using_pred_data_dict[args.base_model]["pred_base_cpu_util"] = tmp_using_pred_data_dict[args.base_model][
                                                            '%s_pred_usage' % args.base_model] / tmp_using_pred_data_dict[
                        args.base_model]['%s_request' % args.base_model] * 100

                    tmp_using_pred_data_dict[args.base_model]["%s_qps_hat" % args.base_model] = \
                    tmp_using_pred_data_dict[args.base_model][
                        "%s_input" % args.base_model].copy()
                    tmp_using_pred_data_dict[args.base_model]["%s_qps_true" % args.base_model] = \
                        tmp_using_pred_data_dict[args.base_model][
                            "%s_true_qps" % args.base_model].copy()

                    tmp_using_pred_data_dict[args.base_model]["%s_total_pred_qps" % args.base_model] = \
                    tmp_using_pred_data_dict[args.base_model][
                        "%s_input" % args.base_model] * tmp_using_pred_data_dict[args.base_model][
                        "%s_pred_replicas" % args.base_model]

                    tmp_using_pred_data_dict[args.base_model]["%s_total_true_qps" % args.base_model] = \
                        tmp_using_pred_data_dict[args.base_model][
                            "%s_true_qps" % args.base_model] * tmp_using_pred_data_dict[args.base_model][
                            "%s_pred_replicas" % args.base_model]

                    tmp_using_pred_data_dict[args.base_model]['%s_total_pred_usage'
                        % args.base_model] = tmp_using_pred_data_dict[args.base_model]['%s_pred_usage'
                                     % args.base_model] * tmp_using_pred_data_dict[args.base_model]['%s_pred_replicas'
                                     % args.base_model]

                    tmp_using_pred_data_dict[args.base_model]["base_RUE"] = tmp_using_pred_data_dict[args.base_model]["RUE"].copy()
                    tmp_using_pred_data_dict[args.base_model]["base_cpu_util"] = tmp_using_pred_data_dict[args.base_model]["cpu_util__pct"].copy()


                    for m in tmp_other_models:
                        if m not in result_config[now_s]["heter"].keys():
                            result_config[now_s]["heter"][m] = {}

                        tmp_using_pred_data_dict[m] = tmp_using_pred_data_dict[m].reset_index(drop=True)
                        tmp_results = usage_predictor.predict_usage(future_qps=tmp_using_pred_data_dict[m][[
                            "%s_input" % m]
                        ], reg=args.reg)

                        tmp_using_pred_data_dict[m] = tmp_using_pred_data_dict[m].reset_index(drop=True)
                        tmp_using_pred_data_dict[m]['%s_pred_usage' % m] = pd.Series(tmp_results)

                        for regg in usage_pred_regs:
                            tmp_true_results = usage_predictor.predict_usage(
                                future_qps=tmp_using_pred_data_dict[m][[
                                    "%s_true_qps" % m]
                                ], reg=regg)

                            tmp_using_pred_data_dict[m][
                                '%s_true_qps_usage_%s' % (m, regg)] = pd.Series(
                                tmp_true_results)

                            tmp_reg_results = usage_predictor.predict_usage(
                                future_qps=tmp_using_pred_data_dict[m][[
                                    "%s_input" % m]
                                ], reg=regg)

                            tmp_using_pred_data_dict[m][
                                '%s_pred_usage_%s' % (m, regg)] = pd.Series(
                                tmp_reg_results)

                        tmp_using_pred_data_dict[m] = pd.merge(tmp_using_pred_data_dict[m],
                                                               tmp_using_pred_data_dict[args.base_model][[
                        "unit","pred_base_cpu_util","base_RUE","base_cpu_util"
                        ]],on='unit',how='inner').reset_index(drop=True)
                        pred_maps = perf_maps[now_s][m].predict_ll(tmp_using_pred_data_dict[m][["pred_base_cpu_util"]].to_numpy())
                        pred_maps2 = perf_maps[now_s][m].predict_ll(tmp_using_pred_data_dict[m][["base_cpu_util"]].to_numpy())

                        tmp_using_pred_data_dict[m]['%s_pred_map' % m] = pd.Series(pred_maps)
                        tmp_using_pred_data_dict[m]['%s_pred_map2' % m] = pd.Series(pred_maps2)



                        tmp_using_pred_data_dict[m]['%s_pred_usage' % m] = tmp_using_pred_data_dict[m]['%s_pred_usage' % m] /  tmp_using_pred_data_dict[m]['%s_pred_map' % m]

                        tmp_using_pred_data_dict[m]['%s_pred_usage2' % m] = tmp_using_pred_data_dict[m]['%s_pred_usage' % m] / \
                                                                   tmp_using_pred_data_dict[m]['%s_pred_map2' % m]

                        tmp_using_pred_data_dict[m]['%s_total_pred_usage'
                                                          % m] = tmp_using_pred_data_dict[m][
                                                                                   '%s_pred_usage'
                                                                                   % m] * tmp_using_pred_data_dict[m][
                                                                                   '%s_pred_replicas'
                                                                                   % m]

                        tmp_using_pred_data_dict[m]['%s_total_pred_usage2'
                                            % m] = tmp_using_pred_data_dict[m][
                                                       '%s_pred_usage2'
                                                       % m] * tmp_using_pred_data_dict[m][
                                                       '%s_pred_replicas'
                                                       % m]

                        for regg in usage_pred_regs:
                            tmp_using_pred_data_dict[m][
                                '%s_true_qps_usage_%s' % (m, regg)] = tmp_using_pred_data_dict[m]['%s_true_qps_usage_%s' % (m,regg)] / tmp_using_pred_data_dict[m]['%s_pred_map' % m]
                            tmp_using_pred_data_dict[m][
                                '%s_pred_usage_%s' % (m, regg)] = tmp_using_pred_data_dict[m][
                                                                          '%s_pred_usage_%s' % (m, regg)] / \
                                                                      tmp_using_pred_data_dict[m]['%s_pred_map' % m]

                            tmp_using_pred_data_dict[m]['%s_total_true_qps_usage_%s'
                                                        % (m, regg)] = \
                                tmp_using_pred_data_dict[m][
                                    '%s_true_qps_usage_%s' % (m, regg)] * \
                                tmp_using_pred_data_dict[m]['%s_pred_replicas'
                                                            % m]

                            tmp_using_pred_data_dict[m]['%s_total_pred_usage_%s'
                                                        % (m, regg)] = \
                                tmp_using_pred_data_dict[m][
                                    '%s_pred_usage_%s' % (m, regg)] * \
                                tmp_using_pred_data_dict[m]['%s_pred_replicas'
                                                            % m]


                        tmp_using_pred_data_dict[m]["%s_qps_hat" % m] = tmp_using_pred_data_dict[m][
                            "%s_input" % m].copy()

                        tmp_using_pred_data_dict[m]["%s_qps_true" % m] = tmp_using_pred_data_dict[m][
                            "%s_true_qps" % m].copy()

                        tmp_using_pred_data_dict[m]["%s_total_pred_qps" % m] = tmp_using_pred_data_dict[m][
                                                                                   "%s_input" % m] * \
                                                                               tmp_using_pred_data_dict[m][
                                                                                   "%s_pred_replicas" % m]

                        tmp_using_pred_data_dict[m]["%s_total_true_qps" % m] = tmp_using_pred_data_dict[m][
                                                                                   "%s_true_qps" % m] * \
                                                                               tmp_using_pred_data_dict[m][
                                                                                   "%s_pred_replicas" % m]

                    print(result_config[now_s])
                    result_config[now_s]["qps"] = {}
                    result_config[now_s]["usage"] = {}

                    out_data = test_data[["ds","hour","unit","total_qps",args.qps_pred_col,"y","total_usage","num_containers","total_quota","request"]].dropna().reset_index(drop=True)
                    out_data = out_data[(out_data.hour >= (final_stat_hour+kk))
                                        & (out_data.hour < (final_stat_hour+kk+t_step))].dropna().reset_index(drop=True)
                    out_data["qps_hat"] = out_data[args.qps_pred_col] / out_data["num_containers"]
                    out_data["qps_true"] = out_data["y"] / out_data["num_containers"]
                    out_data["total_pred_qps"] = out_data[args.qps_pred_col].copy()
                    out_data["total_true_qps"] = out_data["y"].copy()
                    out_data["ground_usage"] = out_data["total_usage"] / out_data["num_containers"]

                    # "ds",args.sample_time,
                    out_data = pd.merge(out_data,tmp_using_pred_data_dict[args.base_model][[args.sample_time,
                    "unit","%s_total_qps" % args.base_model,
                    "%s_total_ground_qps" % args.base_model,"%s_ground_qps" % args.base_model,
                    "%s_qps_hat" % args.base_model,"%s_qps_true" % args.base_model,
                    "%s_total_true_qps" % args.base_model,"%s_total_pred_qps" % args.base_model,"%s_total_usage" % args.base_model,
                        "%s_total_ground_usage" % args.base_model,
                    "%s_total_pred_usage" % args.base_model,"%s_ground_usage" % args.base_model,"%s_pred_usage" % args.base_model,
                    "%s_replicas" % args.base_model,"%s_pred_replicas" % args.base_model,"%s_request" % args.base_model,
                    "%s_pred_quota" % args.base_model,"pred_base_cpu_util","base_RUE"
                                    ]+ ["%s_true_qps_usage_%s" % (args.base_model, jki)
                                                                                       for jki in usage_pred_regs] +
                                                                                  ["%s_pred_usage_%s" % (args.base_model, jki) for jki
                                                                                   in usage_pred_regs] + [
                                                                                      "%s_total_true_qps_usage_%s" % (
                                                                                          args.base_model, jki) for jki in
                                                                                      usage_pred_regs] +
                                                                                  ["%s_total_pred_usage_%s" % (args.base_model, jki)
                                                                                   for jki in usage_pred_regs]],on=['unit'],how='right').reset_index(drop=True)
                    # 'ds', ,args.sample_time

                    out_data["base_ground_RUE"] = out_data["%s_ground_usage" % args.base_model]*1000 / out_data["%s_ground_qps" % args.base_model]

                    print(out_data.head())

                    out_data = out_data.fillna(0)

                    for m in tmp_other_models:
                        tmp_using_pred_data_dict[m]["%s_RUE" % m] = tmp_using_pred_data_dict[m]["RUE"].copy()
                        tmp_using_pred_data_dict[m]["%s_ground_RUE" % m] = tmp_using_pred_data_dict[m]["%s_ground_usage" % m] * 1000 / tmp_using_pred_data_dict[m]["%s_ground_qps" % m]

                        tmp_using_pred_data_dict[m]["%s_map" % m] = tmp_using_pred_data_dict[m]["base_RUE"] / tmp_using_pred_data_dict[m]["RUE"]

                        out_data = pd.merge(out_data, tmp_using_pred_data_dict[m][[
                                                                                      "unit", "%s_total_qps" % m,
                                                                                              "%s_total_ground_qps" % m,
                                                                                              "%s_ground_qps" % m,
                                                                                              "%s_qps_hat" % m,
                                                                                              "%s_qps_true" % m,
                                                                                              "%s_total_true_qps" % m,
                                                                                              "%s_total_pred_qps" % m,
                                                                                              "%s_total_usage" % m,
                                                                                              "%s_total_ground_usage" % m,
                                                                                              "%s_total_pred_usage" % m,
                                                                                              "%s_total_pred_usage2" % m,
                                                                                              "%s_ground_usage" % m,
                                                                                              "%s_pred_usage" % m,
                                                                                              "%s_pred_usage2" % m,
                                                                                              "%s_map" % m,
                                                                                              "%s_pred_map" % m,
                                                                                              "%s_pred_map2" % m,
                                                                                              "%s_replicas" % m,
                                                                                              "%s_pred_replicas" % m,
                                                                                              "%s_request" % m,
                                                                                              "%s_pred_quota" % m,
                                                                                              "%s_RUE" % m,
                                                                                              "%s_ground_RUE" % m
                                                                                  ] + ["%s_true_qps_usage_%s" % (m, jki)
                                                                                       for jki in usage_pred_regs] +
                                                                                  ["%s_pred_usage_%s" % (m, jki) for jki
                                                                                   in usage_pred_regs] + [
                                                                                      "%s_total_true_qps_usage_%s" % (
                                                                                      m, jki) for jki in
                                                                                      usage_pred_regs] +
                                                                                  ["%s_total_pred_usage_%s" % (m, jki)
                                                                                   for jki in usage_pred_regs]],
                                            on=['unit'], how='left').reset_index(drop=True)


                        # "ds",args.sample_time, 'ds',args.sample_time
                        out_data["%s_ground_map" % m] = out_data["base_ground_RUE"] / out_data["%s_ground_RUE" % m]

                        tmp_out_data = out_data[["%s_map" % m,'%s_pred_map' % m]].dropna()
                        tmp_out_data2 = out_data[["%s_map" % m,'%s_pred_map2' % m]].dropna()

                        tmp_out_data3 = out_data[["%s_ground_map" % m, '%s_pred_map' % m]].dropna()
                        tmp_out_data4 = out_data[["%s_ground_map" % m, '%s_pred_map2' % m]].dropna()

                        try:
                            result_config[now_s]["heter"][m]["mape"] = utils.MAPE(
                            y_true=tmp_out_data["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data['%s_pred_map' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mape2"] = utils.MAPE(
                            y_true=tmp_out_data2["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data2['%s_pred_map2' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mape3"] = utils.MAPE(
                            y_true=tmp_out_data3["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data3['%s_pred_map' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mape4"] = utils.MAPE(
                            y_true=tmp_out_data4["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data4['%s_pred_map2' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mse"] = MSE(
                            y_true=tmp_out_data["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data['%s_pred_map' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mse2"] = MSE(
                            y_true=tmp_out_data2["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data2['%s_pred_map2' % m].to_numpy())
                        except Exception as e:
                            print(e)
                        try:
                            result_config[now_s]["heter"][m]["mse3"] = MSE(
                            y_true=tmp_out_data3["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data3['%s_pred_map' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mse4"] = MSE(
                            y_true=tmp_out_data4["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data4['%s_pred_map2' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mae"] = MAE(
                            y_true=tmp_out_data["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data['%s_pred_map' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mae2"] = MAE(
                            y_true=tmp_out_data2["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data2['%s_pred_map2' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mae3"] = MAE(
                            y_true=tmp_out_data3["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data3['%s_pred_map' % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["heter"][m]["mae4"] = MAE(
                            y_true=tmp_out_data4["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data4['%s_pred_map2' % m].to_numpy())
                        except Exception as e:
                            print(e)

                    out_data = out_data.fillna(0)

                    print(result_config[now_s])

                    out_data = out_data.reset_index(drop=True)
                    result_config[now_s]['qps'][args.base_model] = {}
                    result_config[now_s]['qps']['total'] = {}

                    try:
                        result_config[now_s]['qps'][args.base_model]["mape"] = utils.MAPE(
                            y_true=out_data["%s_total_ground_qps" % args.base_model].to_numpy(),
                            y_pred=out_data["%s_total_pred_qps" % args.base_model].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]['qps']['total']["mape"] = utils.MAPE(y_true=out_data["total_qps"].to_numpy(),
                                                                           y_pred=out_data["total_pred_qps"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]['qps'][args.base_model]["mse"] = MSE(
                            y_true=out_data["%s_total_ground_qps" % args.base_model].to_numpy(),
                            y_pred=out_data["%s_total_pred_qps" % args.base_model].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]['qps']['total']["mse"] = MSE(y_true=out_data["total_qps"].to_numpy(),
                                                                   y_pred=out_data["total_pred_qps"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]['qps'][args.base_model]["mae"] = MAE(
                            y_true=out_data["%s_total_ground_qps" % args.base_model].to_numpy(),
                            y_pred=out_data["%s_total_pred_qps" % args.base_model].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]['qps']['total']["mae"] = MAE(y_true=out_data["total_qps"].to_numpy(),
                                                                   y_pred=out_data["total_pred_qps"].to_numpy())
                    except Exception as e:
                        print(e)

                    for m in tmp_other_models:
                        result_config[now_s]["qps"][m] = {}
                        try:
                            result_config[now_s]['qps'][m]["mse"] = MSE(
                                y_true=out_data["%s_total_ground_qps" % m].to_numpy(),
                                y_pred=out_data["%s_total_pred_qps" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]['qps'][m]["mae"] = MAE(
                                y_true=out_data["%s_total_ground_qps" % m].to_numpy(),
                                y_pred=out_data["%s_total_pred_qps" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]['qps'][m]["mape"] = utils.MAPE(
                                y_true=out_data["%s_total_ground_qps" % m].to_numpy(),
                                y_pred=out_data["%s_total_pred_qps" % m].to_numpy())
                        except Exception as e:
                            print(e)

                    out_data['total_pred_usage'] = out_data["%s_total_pred_usage" % args.base_model].copy()
                    out_data['total_pred_usage2'] = out_data["%s_total_pred_usage" % args.base_model].copy()

                    for regg in usage_pred_regs:
                        out_data['total_pred_usage_%s' % regg] = out_data["%s_total_pred_usage_%s" % (args.base_model,regg)].copy()
                        out_data['total_true_qps_usage_%s' % regg] = out_data[
                            "%s_total_true_qps_usage_%s" % (args.base_model, regg)].copy()

                    out_data['total_ground_usage'] = out_data["%s_total_ground_usage" % args.base_model].copy()
                    out_data['total_replicas'] = out_data["%s_replicas" % args.base_model].copy()

                    out_data["total_pred_replicas"] = out_data["%s_pred_replicas" % args.base_model].copy()
                    out_data["total_pred_quota"] = out_data["%s_pred_quota" % args.base_model].copy()

                    result_config[now_s]["usage"][args.base_model] = {}

                    try:
                        result_config[now_s]["usage"][args.base_model]["mape"] = utils.MAPE(
                            y_true=out_data["%s_total_ground_usage" % args.base_model].to_numpy(),
                            y_pred=out_data["%s_total_pred_usage" % args.base_model].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"][args.base_model]["mse"] = MSE(
                            y_true=out_data["%s_total_ground_usage" % args.base_model].to_numpy(),
                            y_pred=out_data["%s_total_pred_usage" % args.base_model].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"][args.base_model]["mae"] = MAE(
                            y_true=out_data["%s_total_ground_usage" % args.base_model].to_numpy(),
                            y_pred=out_data["%s_total_pred_usage" % args.base_model].to_numpy())
                    except Exception as e:
                        print(e)

                    for m in tmp_other_models:
                        result_config[now_s]["usage"][m] = {}
                        out_data['total_pred_usage'] = out_data['total_pred_usage'] + out_data[
                            '%s_total_pred_usage' % m]
                        out_data['total_pred_usage2'] = out_data['total_pred_usage2'] + out_data[
                            '%s_total_pred_usage2' % m]

                        out_data['total_ground_usage'] = out_data['total_ground_usage'] + out_data[
                            '%s_total_ground_usage' % m]
                        out_data['total_replicas'] = out_data['total_replicas'] + out_data['%s_replicas' % m]
                        out_data["total_pred_replicas"] = out_data["total_pred_replicas"] + out_data[
                            "%s_pred_replicas" % m]

                        out_data["total_pred_quota"] = out_data["total_pred_quota"] + out_data[
                            "%s_pred_quota" % m]

                        for regg in usage_pred_regs:
                            out_data["total_pred_usage_%s" % regg] = out_data[
                                "%s_total_pred_usage_%s" % (m, regg)] + out_data["total_pred_usage_%s" % regg]
                            out_data["total_true_qps_usage_%s" % regg] = out_data[
                                "%s_total_true_qps_usage_%s" % (m, regg)] + out_data["total_true_qps_usage_%s" % regg]

                        tmp_out_data = out_data[["%s_total_ground_usage" % m, "%s_total_pred_usage" % m]].dropna()
                        tmp_out_data2 = out_data[["%s_total_ground_usage" % m, "%s_total_pred_usage2" % m]].dropna()

                        try:
                            result_config[now_s]["usage"][m]["mape"] = utils.MAPE(
                                y_true=tmp_out_data["%s_total_ground_usage" % m].to_numpy(),
                                y_pred=tmp_out_data["%s_total_pred_usage" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["usage"][m]["mape2"] = utils.MAPE(
                                y_true=tmp_out_data2["%s_total_ground_usage" % m].to_numpy(),
                                y_pred=tmp_out_data2["%s_total_pred_usage2" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["usage"][m]["mae"] = MAE(
                                y_true=tmp_out_data["%s_total_ground_usage" % m].to_numpy(),
                                y_pred=tmp_out_data["%s_total_pred_usage" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["usage"][m]["mae2"] = MAE(
                                y_true=tmp_out_data2["%s_total_ground_usage" % m].to_numpy(),
                                y_pred=tmp_out_data2["%s_total_pred_usage2" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["usage"][m]["mse"] = MSE(
                                y_true=tmp_out_data["%s_total_ground_usage" % m].to_numpy(),
                                y_pred=tmp_out_data["%s_total_pred_usage" % m].to_numpy())
                        except Exception as e:
                            print(e)

                        try:
                            result_config[now_s]["usage"][m]["mse2"] = MSE(
                                y_true=tmp_out_data2["%s_total_ground_usage" % m].to_numpy(),
                                y_pred=tmp_out_data2["%s_total_pred_usage2" % m].to_numpy())
                        except Exception as e:
                            print(e)

                    result_config[now_s]["usage"]["total"] = {}

                    tmp_out_data = out_data[["total_ground_usage", "total_pred_usage"]].dropna()
                    tmp_out_data2 = out_data[["total_ground_usage", "total_pred_usage2"]].dropna()

                    try:
                        result_config[now_s]["usage"]["total"]["mape"] = utils.MAPE(
                            y_true=tmp_out_data["total_ground_usage"].to_numpy(),
                            y_pred=tmp_out_data["total_pred_usage"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"]["total"]["mae"] = MAE(
                            y_true=tmp_out_data["total_ground_usage"].to_numpy(),
                            y_pred=tmp_out_data["total_pred_usage"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"]["total"]["mse"] = MSE(
                            y_true=tmp_out_data["total_ground_usage"].to_numpy(),
                            y_pred=tmp_out_data["total_pred_usage"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"]["total"]["mape2"] = utils.MAPE(
                            y_true=tmp_out_data2["total_ground_usage"].to_numpy(),
                            y_pred=tmp_out_data2["total_pred_usage2"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"]["total"]["mae2"] = MAE(
                            y_true=tmp_out_data2["total_ground_usage"].to_numpy(),
                            y_pred=tmp_out_data2["total_pred_usage2"].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config[now_s]["usage"]["total"]["mse2"] = MSE(
                            y_true=tmp_out_data2["total_ground_usage"].to_numpy(),
                            y_pred=tmp_out_data2["total_pred_usage2"].to_numpy())
                    except Exception as e:
                        print(e)

                    print("out data shape is:")
                    print(out_data.shape)
                    print("out_data column is:")
                    print(out_data.columns)

                    for m in tmp_other_models:
                        final_other_models.append(m)

                    if total_out_data.shape[0] < 1:
                        total_out_data = out_data.copy()
                    else:
                        total_out_data = pd.concat([total_out_data,out_data.copy()],axis=0).reset_index(drop=True)

                final_other_models = list(set(final_other_models))
                other_models = final_other_models[:]
                print("get total out data results")
                result_config["total"]["heter"] = {}
                result_config["total"]["usage"] = {}
                result_config["total"]["qps"] = {}
                for m in other_models:
                    result_config["total"]["heter"][m] = {}

                    tmp_out_data = total_out_data[["%s_map" % m, '%s_pred_map' % m]].dropna()
                    tmp_out_data2 = total_out_data[["%s_map" % m, '%s_pred_map2' % m]].dropna()

                    tmp_out_data3 = total_out_data[["%s_ground_map" % m, '%s_pred_map' % m]].dropna()
                    tmp_out_data4 = total_out_data[["%s_ground_map" % m, '%s_pred_map2' % m]].dropna()

                    try:
                        result_config["total"]["heter"][m]["mape"] = utils.MAPE(
                            y_true=tmp_out_data["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data['%s_pred_map' % m].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mape2"] = utils.MAPE(
                            y_true=tmp_out_data2["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data2['%s_pred_map2' % m].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mape3"] = utils.MAPE(
                            y_true=tmp_out_data3["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data3['%s_pred_map' % m].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mape4"] = utils.MAPE(
                            y_true=tmp_out_data4["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data4['%s_pred_map2' % m].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mse"] = MSE(
                            y_true=tmp_out_data["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data['%s_pred_map' % m].to_numpy())

                        result_config["total"]["heter"][m]["mse_norm"] = MSE(
                            y_true=((tmp_out_data["%s_map" % m]-tmp_out_data["%s_map" % m].mean())/tmp_out_data["%s_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data['%s_pred_map' % m]-tmp_out_data["%s_map" % m].mean())/tmp_out_data["%s_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mse2"] = MSE(
                            y_true=tmp_out_data2["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data2['%s_pred_map2' % m].to_numpy())

                        result_config["total"]["heter"][m]["mse_norm2"] = MSE(
                            y_true=((tmp_out_data2["%s_map" % m] - tmp_out_data2["%s_map" % m].mean()) / tmp_out_data2[
                                "%s_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data2['%s_pred_map2' % m] - tmp_out_data2["%s_map" % m].mean()) /
                                    tmp_out_data2["%s_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mse3"] = MSE(
                            y_true=tmp_out_data3["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data3['%s_pred_map' % m].to_numpy())

                        result_config["total"]["heter"][m]["mse_norm3"] = MSE(
                            y_true=((tmp_out_data3["%s_ground_map" % m] - tmp_out_data3["%s_ground_map" % m].mean())/tmp_out_data3["%s_ground_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data3['%s_pred_map' % m] - tmp_out_data3["%s_ground_map" % m].mean())/tmp_out_data3["%s_ground_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mse4"] = MSE(
                            y_true=tmp_out_data4["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data4['%s_pred_map2' % m].to_numpy())

                        result_config["total"]["heter"][m]["mse_norm4"] = MSE(
                            y_true=((tmp_out_data4["%s_ground_map" % m] - tmp_out_data4["%s_ground_map" % m].mean()) /
                                    tmp_out_data4["%s_ground_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data4['%s_pred_map2' % m] - tmp_out_data4["%s_ground_map" % m].mean()) /
                                    tmp_out_data4["%s_ground_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mae"] = MAE(
                            y_true=tmp_out_data["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data['%s_pred_map' % m].to_numpy())

                        result_config["total"]["heter"][m]["mae_norm"] = MAE(
                            y_true=((tmp_out_data["%s_map" % m] - tmp_out_data["%s_map" % m].mean()) / tmp_out_data[
                                "%s_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data['%s_pred_map' % m] - tmp_out_data["%s_map" % m].mean()) /
                                    tmp_out_data["%s_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mae2"] = MAE(
                            y_true=tmp_out_data2["%s_map" % m].to_numpy(),
                            y_pred=tmp_out_data2['%s_pred_map2' % m].to_numpy())

                        result_config["total"]["heter"][m]["mae_norm2"] = MAE(
                            y_true=((tmp_out_data2["%s_map" % m] - tmp_out_data2["%s_map" % m].mean()) / tmp_out_data2[
                                "%s_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data2['%s_pred_map2' % m] - tmp_out_data2["%s_map" % m].mean()) /
                                    tmp_out_data2["%s_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mae3"] = MAE(
                            y_true=tmp_out_data3["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data3['%s_pred_map' % m].to_numpy())

                        result_config["total"]["heter"][m]["mae_norm3"] = MAE(
                            y_true=((tmp_out_data3["%s_ground_map" % m] - tmp_out_data3["%s_ground_map" % m].mean()) /
                                    tmp_out_data3["%s_ground_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data3['%s_pred_map' % m] - tmp_out_data3["%s_ground_map" % m].mean()) /
                                    tmp_out_data3["%s_ground_map" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["heter"][m]["mae4"] = MAE(
                            y_true=tmp_out_data4["%s_ground_map" % m].to_numpy(),
                            y_pred=tmp_out_data4['%s_pred_map2' % m].to_numpy())

                        result_config["total"]["heter"][m]["mae_norm4"] = MAE(
                            y_true=((tmp_out_data4["%s_ground_map" % m] - tmp_out_data4["%s_ground_map" % m].mean()) /
                                    tmp_out_data4["%s_ground_map" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data4['%s_pred_map2' % m] - tmp_out_data4["%s_ground_map" % m].mean()) /
                                    tmp_out_data4["%s_ground_map" % m].std()).to_numpy())


                    except Exception as e:
                        print(e)

                print(result_config['total'])

                total_out_data = total_out_data.reset_index(drop=True)
                result_config["total"]['qps'][args.base_model] = {}
                result_config["total"]['qps']['total'] = {}

                try:
                    result_config["total"]['qps'][args.base_model]["mape"] = utils.MAPE(
                        y_true=total_out_data["%s_total_ground_qps" % args.base_model].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_qps" % args.base_model].to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]['qps']['total']["mape"] = utils.MAPE(y_true=total_out_data["total_qps"].to_numpy(),
                                                                          y_pred=total_out_data["total_pred_qps"].to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]['qps'][args.base_model]["mse"] = MSE(
                        y_true=total_out_data["%s_total_ground_qps" % args.base_model].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_qps" % args.base_model].to_numpy())

                    result_config["total"]['qps'][args.base_model]["mse_norm"] = MSE(
                        y_true=((total_out_data["%s_total_ground_qps" % args.base_model]-total_out_data[
                            "%s_total_ground_qps" % args.base_model].mean())/total_out_data["%s_total_ground_qps" % args.base_model].std()).to_numpy(),
                        y_pred=((total_out_data["%s_total_pred_qps" % args.base_model]-total_out_data[
                            "%s_total_ground_qps" % args.base_model].mean())/total_out_data["%s_total_ground_qps" % args.base_model].std()).to_numpy())
                #     total_out_data["%s_total_pred_qps" % args.base_model]
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]['qps']['total']["mse"] = MSE(y_true=total_out_data["total_qps"].to_numpy(),
                                                                  y_pred=total_out_data["total_pred_qps"].to_numpy())

                    result_config["total"]['qps']['total']["mse_norm"] = MSE(
                        y_true=((total_out_data["total_qps" ] - total_out_data[
                            'total_qps'].mean()) / total_out_data[
                                    'total_qps'].std()).to_numpy(),
                        y_pred=((total_out_data["total_pred_qps" ] - total_out_data[
                            'total_qps'].mean()) / total_out_data[
                                    'total_qps'].std()).to_numpy())

                except Exception as e:
                    print(e)

                try:
                    result_config["total"]['qps'][args.base_model]["mae"] = MAE(
                    y_true=total_out_data["%s_total_ground_qps" % args.base_model].to_numpy(),
                    y_pred=total_out_data["%s_total_pred_qps" % args.base_model].to_numpy())

                    result_config["total"]['qps'][args.base_model]["mae_norm"] = MAE(
                        y_true=((total_out_data["%s_total_ground_qps" % args.base_model] - total_out_data[
                            "%s_total_ground_qps" % args.base_model].mean()) / total_out_data[
                                    "%s_total_ground_qps" % args.base_model].std()).to_numpy(),
                        y_pred=((total_out_data["%s_total_pred_qps" % args.base_model] - total_out_data[
                            "%s_total_ground_qps" % args.base_model].mean()) / total_out_data[
                                    "%s_total_ground_qps" % args.base_model].std()).to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]['qps']['total']["mae"] = MAE(y_true=total_out_data["total_qps"].to_numpy(),
                                                                  y_pred=total_out_data["total_pred_qps"].to_numpy())

                    result_config["total"]['qps']['total']["mae_norm"] = MAE(
                        y_true=((total_out_data["total_qps"] - total_out_data[
                            'total_qps'].mean()) / total_out_data[
                                    'total_qps'].std()).to_numpy(),
                        y_pred=((total_out_data["total_pred_qps"] - total_out_data[
                            'total_qps'].mean()) / total_out_data[
                                    'total_qps'].std()).to_numpy())
                except Exception as e:
                    print(e)

                for m in other_models:
                    result_config["total"]["qps"][m] = {}
                    try:
                        result_config["total"]['qps'][m]["mse"] = MSE(
                        y_true=total_out_data["%s_total_ground_qps" % m].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_qps" % m].to_numpy())

                        result_config["total"]['qps'][m]["mse_norm"] = MSE(
                            y_true=((total_out_data["%s_total_ground_qps" % m] - total_out_data[
                                "%s_total_ground_qps" % m].mean()) / total_out_data[
                                        "%s_total_ground_qps" % m].std()).to_numpy(),
                            y_pred=((total_out_data["%s_total_pred_qps" % m] - total_out_data[
                                "%s_total_ground_qps" % m].mean()) / total_out_data[
                                        "%s_total_ground_qps" % m].std()).to_numpy())


                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]['qps'][m]["mae"] = MAE(
                        y_true=total_out_data["%s_total_ground_qps" % m].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_qps" % m].to_numpy())

                        result_config["total"]['qps'][m]["mae_norm"] = MAE(
                            y_true=((total_out_data["%s_total_ground_qps" % m] - total_out_data[
                                "%s_total_ground_qps" % m].mean()) / total_out_data[
                                        "%s_total_ground_qps" % m].std()).to_numpy(),
                            y_pred=((total_out_data["%s_total_pred_qps" % m] - total_out_data[
                                "%s_total_ground_qps" % m].mean()) / total_out_data[
                                        "%s_total_ground_qps" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]['qps'][m]["mape"] = utils.MAPE(
                        y_true=total_out_data["%s_total_ground_qps" % m].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_qps" % m].to_numpy())
                    except Exception as e:
                        print(e)


                result_config["total"]["usage"][args.base_model] = {}

                try:
                    result_config["total"]["usage"][args.base_model]["mape"] = utils.MAPE(
                        y_true=total_out_data["%s_total_ground_usage" % args.base_model].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_usage" % args.base_model].to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"][args.base_model]["mse"] = MSE(
                        y_true=total_out_data["%s_total_ground_usage" % args.base_model].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_usage" % args.base_model].to_numpy())

                    result_config["total"]["usage"][args.base_model]["mse_norm"] = MSE(
                        y_true=((total_out_data["%s_total_ground_usage" % args.base_model]-total_out_data[
                            "%s_total_ground_usage" % args.base_model].mean())/total_out_data[
                            "%s_total_ground_usage" % args.base_model].std()).to_numpy(),
                        y_pred=((total_out_data["%s_total_pred_usage" % args.base_model]-total_out_data[
                            "%s_total_ground_usage" % args.base_model].mean())/total_out_data[
                            "%s_total_ground_usage" % args.base_model].std()).to_numpy())
                #     total_out_data["%s_total_pred_usage" % args.base_model]
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"][args.base_model]["mae"] = MAE(
                        y_true=total_out_data["%s_total_ground_usage" % args.base_model].to_numpy(),
                        y_pred=total_out_data["%s_total_pred_usage" % args.base_model].to_numpy())

                    result_config["total"]["usage"][args.base_model]["mae_norm"] = MAE(
                        y_true=((total_out_data["%s_total_ground_usage" % args.base_model] - total_out_data[
                            "%s_total_ground_usage" % args.base_model].mean()) / total_out_data[
                                    "%s_total_ground_usage" % args.base_model].std()).to_numpy(),
                        y_pred=((total_out_data["%s_total_pred_usage" % args.base_model] - total_out_data[
                            "%s_total_ground_usage" % args.base_model].mean()) / total_out_data[
                                    "%s_total_ground_usage" % args.base_model].std()).to_numpy())
                except Exception as e:
                    print(e)

                for m in other_models:
                    result_config["total"]["usage"][m] = {}

                    tmp_out_data = total_out_data[["%s_total_ground_usage" % m, "%s_total_pred_usage" % m]].dropna()
                    tmp_out_data2 = total_out_data[["%s_total_ground_usage" % m, "%s_total_pred_usage2" % m]].dropna()


                    try:
                        result_config["total"]["usage"][m]["mape"] = utils.MAPE(
                            y_true=tmp_out_data["%s_total_ground_usage" % m].to_numpy(),
                            y_pred=tmp_out_data["%s_total_pred_usage" % m].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["usage"][m]["mape2"] = utils.MAPE(
                            y_true=tmp_out_data2["%s_total_ground_usage" % m].to_numpy(),
                            y_pred=tmp_out_data2["%s_total_pred_usage2" % m].to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["usage"][m]["mae"] = MAE(
                            y_true=tmp_out_data["%s_total_ground_usage" % m].to_numpy(),
                            y_pred=tmp_out_data["%s_total_pred_usage" % m].to_numpy())

                        result_config["total"]["usage"][m]["mae_norm"] = MAE(
                            y_true=((tmp_out_data["%s_total_ground_usage" % m]-tmp_out_data["%s_total_ground_usage" % m].mean())/tmp_out_data["%s_total_ground_usage" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data["%s_total_pred_usage" % m]-tmp_out_data["%s_total_ground_usage" % m].mean())/tmp_out_data["%s_total_ground_usage" % m].std()).to_numpy())
                    #     tmp_out_data["%s_total_pred_usage" % m]
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["usage"][m]["mae2"] = MAE(
                            y_true=tmp_out_data2["%s_total_ground_usage" % m].to_numpy(),
                            y_pred=tmp_out_data2["%s_total_pred_usage2" % m].to_numpy())

                        result_config["total"]["usage"][m]["mae2_norm"] = MAE(
                            y_true=((tmp_out_data2["%s_total_ground_usage" % m] - tmp_out_data2[
                                "%s_total_ground_usage" % m].mean()) / tmp_out_data2[
                                        "%s_total_ground_usage" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data2["%s_total_pred_usage2" % m] - tmp_out_data2[
                                "%s_total_ground_usage" % m].mean()) / tmp_out_data2[
                                        "%s_total_ground_usage" % m].std()).to_numpy())

                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["usage"][m]["mse"] = MSE(
                            y_true=tmp_out_data["%s_total_ground_usage" % m].to_numpy(),
                            y_pred=tmp_out_data["%s_total_pred_usage" % m].to_numpy())

                        result_config["total"]["usage"][m]["mse_norm"] = MSE(
                            y_true=((tmp_out_data["%s_total_ground_usage" % m] - tmp_out_data[
                                "%s_total_ground_usage" % m].mean()) / tmp_out_data[
                                        "%s_total_ground_usage" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data["%s_total_pred_usage" % m] - tmp_out_data[
                                "%s_total_ground_usage" % m].mean()) / tmp_out_data[
                                        "%s_total_ground_usage" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                    try:
                        result_config["total"]["usage"][m]["mse2"] = MSE(
                            y_true=tmp_out_data2["%s_total_ground_usage" % m].to_numpy(),
                            y_pred=tmp_out_data2["%s_total_pred_usage2" % m].to_numpy())

                        result_config["total"]["usage"][m]["mse2_norm"] = MSE(
                            y_true=((tmp_out_data2["%s_total_ground_usage" % m] - tmp_out_data2[
                                "%s_total_ground_usage" % m].mean()) / tmp_out_data2[
                                        "%s_total_ground_usage" % m].std()).to_numpy(),
                            y_pred=((tmp_out_data2["%s_total_pred_usage2" % m] - tmp_out_data2[
                                "%s_total_ground_usage" % m].mean()) / tmp_out_data2[
                                        "%s_total_ground_usage" % m].std()).to_numpy())
                    except Exception as e:
                        print(e)

                result_config["total"]["usage"]["total"] = {}

                tmp_out_data = total_out_data[["total_ground_usage", "total_pred_usage"]].dropna()
                tmp_out_data2 = total_out_data[["total_ground_usage", "total_pred_usage2"]].dropna()

                try:
                    result_config["total"]["usage"]["total"]["mape"] = utils.MAPE(
                    y_true=tmp_out_data["total_ground_usage"].to_numpy(),
                    y_pred=tmp_out_data["total_pred_usage"].to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"]["total"]["mae"] = MAE(
                    y_true=tmp_out_data["total_ground_usage"].to_numpy(),
                    y_pred=tmp_out_data["total_pred_usage"].to_numpy())

                    result_config["total"]["usage"]["total"]["mae_norm"] = MAE(
                        y_true=((tmp_out_data["total_ground_usage"] - tmp_out_data["total_ground_usage"].mean())/tmp_out_data["total_ground_usage"].std()).to_numpy(),
                        y_pred=((tmp_out_data["total_pred_usage"] - tmp_out_data["total_ground_usage"].mean())/tmp_out_data["total_ground_usage"].std()).to_numpy())
                #     tmp_out_data["total_pred_usage"]


                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"]["total"]["mse"] = MSE(
                    y_true=tmp_out_data["total_ground_usage"].to_numpy(),
                    y_pred=tmp_out_data["total_pred_usage"].to_numpy())

                    result_config["total"]["usage"]["total"]["mse_norm"] = MSE(
                        y_true=((tmp_out_data["total_ground_usage"] - tmp_out_data["total_ground_usage"].mean()) /
                                tmp_out_data["total_ground_usage"].std()).to_numpy(),
                        y_pred=((tmp_out_data["total_pred_usage"] - tmp_out_data["total_ground_usage"].mean()) /
                                tmp_out_data["total_ground_usage"].std()).to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"]["total"]["mape2"] = utils.MAPE(
                    y_true=tmp_out_data2["total_ground_usage"].to_numpy(),
                    y_pred=tmp_out_data2["total_pred_usage2"].to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"]["total"]["mae2"] = MAE(
                    y_true=tmp_out_data2["total_ground_usage"].to_numpy(),
                    y_pred=tmp_out_data2["total_pred_usage2"].to_numpy())

                    result_config["total"]["usage"]["total"]["mae2_norm"] = MAE(
                        y_true=((tmp_out_data2["total_ground_usage"] - tmp_out_data2["total_ground_usage"].mean()) /
                                tmp_out_data2["total_ground_usage"].std()).to_numpy(),
                        y_pred=((tmp_out_data2["total_pred_usage2"] - tmp_out_data2["total_ground_usage"].mean()) /
                                tmp_out_data2["total_ground_usage"].std()).to_numpy())
                except Exception as e:
                    print(e)

                try:
                    result_config["total"]["usage"]["total"]["mse2"] = MSE(
                    y_true=tmp_out_data2["total_ground_usage"].to_numpy(),
                    y_pred=tmp_out_data2["total_pred_usage2"].to_numpy())

                    result_config["total"]["usage"]["total"]["mse2_norm"] = MSE(
                        y_true=((tmp_out_data2["total_ground_usage"] - tmp_out_data2["total_ground_usage"].mean()) /
                                tmp_out_data2["total_ground_usage"].std()).to_numpy(),
                        y_pred=((tmp_out_data2["total_pred_usage2"] - tmp_out_data2["total_ground_usage"].mean()) /
                                tmp_out_data2["total_ground_usage"].std()).to_numpy())
                except Exception as e:
                    print(e)

                print("total out data shape is:")
                print(total_out_data.shape)
                print("total out_data column is:")
                print(total_out_data.columns)

                save_config(result_config, os.path.join(app_date_result_path, "result.json"))
                total_out_data.to_csv(os.path.join(app_date_result_path,"predict_results.csv"),index=False)