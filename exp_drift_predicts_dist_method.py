import pandas as pd
import numpy as np
import os
import argparse
import torch
import time
import json
from find_range2 import find_train_range_gmm,find_train_range_dist
from heter_predict_model import UsageMapPredictor
import datetime


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
    aim_apps = [
        "MS0"
    ]


    parser = argparse.ArgumentParser(description='[Heter_Upgradation] AutoScaling')

    # , required=True
    parser.add_argument('--total_mode', type=int, default=0,
                        help='use total data of mean data')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)

    # , required=True
    parser.add_argument('--base_dir', type=str, default="hetergenous_data_exp/pod_metric_data_apps", help='data')
    parser.add_argument('--result_dir', type=str, default="./results", help='data')
    parser.add_argument("--base_model",type=str,default="826X")
    parser.add_argument("--start_date",type=str,default='20220815')
    parser.add_argument("--end_date",type=str,default='20221020')

    parser.add_argument("--init_size",type=int,default=6)
    parser.add_argument("--chunck_size", type=int, default=480)
    parser.add_argument("--unit_step", type=int, default=480, help="")

    parser.add_argument("--freq",type=str,default='t')
    parser.add_argument("--step",type=int,default=1,help='the step when exploring the train data range')
    parser.add_argument("--diff_limit",type=float,default=0.15)
    parser.add_argument("--n_components",type=int,default=0,help="the limit of the n_components")
    parser.add_argument("--select_n",type=str,default='bic',help="the metric for the n_components decision")
    parser.add_argument("--diff_win",type=int,default=3,help="how to judge the convergence of GMM")
    parser.add_argument("--unit",type=str,default="units",help="the col for the time")
    parser.add_argument("--test_size",type=float,default=0.15,help="the size of test data")
    parser.add_argument("--sample_time",type=str,default="sample_time_n")
    parser.add_argument("--norm",type=int,default=0)
    parser.add_argument("--x_unit",type=int,default=1)
    parser.add_argument("--y_unit",type=int,default=1000)
    parser.add_argument("--block_size",type=int,default=6)
    parser.add_argument("--standard",type=int,default=1)
    parser.add_argument("--gmm_mode",type=int,default=0)
    parser.add_argument("--norm_mode",type=int,default=2)
    parser.add_argument("--cov_col",type=str,default='cpu_util__pct')
    parser.add_argument("--pdc_col",type=str,default='RUE')
    parser.add_argument("--exp",type=int,default=4)
    parser.add_argument("--date_size", type=int, default=21)
    parser.add_argument("--iter",type=int,default=3)
    parser.add_argument("--idx", type=int, default=0)



    parser.add_argument("--qps_level_num",type=int,default=24)
    parser.add_argument("--minute_unit",type=int,default=15)
    parser.add_argument("--save_model",type=int,default=0)

    parser.add_argument("--drift_model", type=str, default="context")
    parser.add_argument("--p_val", type=float, default=0.05)
    parser.add_argument("--if_update_ref", type=int, default=0)
    parser.add_argument("--x_ref_preprocessed", type=int, default=0)
    parser.add_argument("--preprocess_at_init", type=int, default=1)
    parser.add_argument("--inverse", type=int, default=1)
    parser.add_argument("--n_permutations", type=int, default=100)
    parser.add_argument("--lambda_rd_max", type=float, default=0.2)
    parser.add_argument("--update_x_ref", type=str, default="reservoir_sampling")
    parser.add_argument("--update_ref", type=str, default="last")
    parser.add_argument("--backend", type=str, default="pytorch")
    parser.add_argument("--prop_c_held", type=float, default=0.25)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--context_cols", type=str, default="minute")
    parser.add_argument("--heter_context", type=int, default=1)
    parser.add_argument("--replicas_col", type=str, default="replicas")
    parser.add_argument("--heter_method", type=str, default="h")
    parser.add_argument("--qps_true_dir",type=str,default="exp_total_qps_data_aggr2/mean")

    args = parser.parse_args()

    base_dir = args.base_dir

    if args.if_update_ref < 1:
        if_update_ref = False
    else:
        if_update_ref = True

    if args.x_ref_preprocessed < 1:
        x_ref_preprocessed = False
    else:
        x_ref_preprocessed = True

    if args.preprocess_at_init < 1:
        preprocess_at_init = False
    else:
        preprocess_at_init = True

    if args.inverse  < 1:
        inverse = False
    else:
        inverse = True

    if args.save_model < 1:
        save_model = False
    else:
        save_model = True

    if args.early_stop < 1:
        early_stop = False
    else:
        early_stop = True

    if args.heter_context < 1:
        heter_context = False
    else:
        heter_context = True

    tmp_context_cols = args.context_cols.strip().split(",")
    context_col = []
    for tcc in tmp_context_cols:
        if tcc:
            context_col.append(tcc)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir,exist_ok=True)

    # time_now =
    exp_res_dir = os.path.join(args.result_dir,str(args.exp))
    if not os.path.exists(exp_res_dir):
        os.makedirs(exp_res_dir,exist_ok=True)

    use_gpu = True if torch.cuda.is_available() and args.use_gpu > 0 else False

    if use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]




    for ap in aim_apps[args.idx:args.idx+1]:
        print(ap)
        app_path = os.path.join(base_dir,ap)
        app_out_res_path = os.path.join(exp_res_dir,ap)
        if not os.path.exists(app_out_res_path):
            os.makedirs(app_out_res_path,exist_ok=True)

        # aim_model_file = []
        aim_models = {}
        app_heter_data = {}
        for j in os.listdir(app_path):
            if '.csv' in j and "C3" not in j:
                # aim_model_file.append(j)
                aim_models[j.split(".")[0]] = j

        aim_date = []

        true_data = pd.read_csv(os.path.join(args.qps_true_dir, "%s.csv" % ap))
        true_data['sample_time_n'] = pd.to_datetime(true_data['sample_time_n'])

        true_data["ds"] = true_data["ds"].astype(int)
        true_data["ds"] = true_data["ds"].astype(str)

        true_data = true_data.sort_values("sample_time_n").reset_index(drop=True)

        for j in aim_models.keys():
            # container_app_group_name" ,"cpu_model" "sample_time__m" "max_cpu_util" "qps_std","rt_std","usage_std",
            app_heter_data[j] = pd.read_csv(os.path.join(app_path,aim_models[j]))
            app_heter_data[j] = app_heter_data[j][["container_app_name"
                                                   ,
                                                   "throughput_type",
                                                   "ds","minute","qps__Q_s",
                                                   "rt__ms_Q","cpu_usage","replicas",
                                                   "cpu_util__pct",
                                                   "request_cpu",args.sample_time,"RUE"]].reset_index(drop=True)


            app_heter_data[j]["minute_raw"] = app_heter_data[j]["minute"].copy()
            app_heter_data[j]["unit"] = app_heter_data[j]["minute_raw"]//5

            # app_heter_data[j][""] =

            app_heter_data[j]["minute"] = app_heter_data[j]["minute"] // args.minute_unit

            app_heter_data[j]['total_qps'] = app_heter_data[j]['qps__Q_s'] * app_heter_data[j]['replicas']
            app_heter_data[j]['total_usage'] = app_heter_data[j]['cpu_usage'] * app_heter_data[j]['replicas']
            app_heter_data[j][args.sample_time] = pd.to_datetime(app_heter_data[j][args.sample_time])

            if args.total_mode > 0:
                app_heter_data[j]['RUE'] = app_heter_data[j]['total_usage']*args.y_unit / app_heter_data[j]['total_qps']
            else:
                app_heter_data[j]['RUE'] = app_heter_data[j]['cpu_usage']*args.y_unit / app_heter_data[j]['qps__Q_s']

            print(app_heter_data[j].shape)

            app_heter_data[j] = app_heter_data[j].dropna().sort_values(args.sample_time).reset_index(drop=True)
            print(app_heter_data[j].shape)
            app_heter_data[j]["ds"] = app_heter_data[j]["ds"].astype(int)
            app_heter_data[j]["ds"] = app_heter_data[j]["ds"].astype(str)

            app_heter_data[j][args.unit] = app_heter_data[j][args.sample_time] - pd.to_datetime("2022-08-01 00:00:00")
            app_heter_data[j][args.unit] = app_heter_data[j][args.unit].astype(np.int64)
            app_heter_data[j][args.unit] = app_heter_data[j][args.unit] // 60000000000
            app_heter_data[j][args.unit] = app_heter_data[j][args.unit].astype(int)

            now_date = app_heter_data[j]["ds"].unique().tolist()
            print(now_date)
            aim_date += now_date
            print(aim_date)

        aim_date = list(set(aim_date))

        aim_date.sort()
        print(aim_date)

        for d in aim_date:
            if d < args.start_date:
                continue
            if d > args.end_date:
                continue
            print("predict date is: %s" % d)
            configs = {}
            configs['base'] = d
            configs["standard"] = args.standard
            configs['dates'] = args.date_size
            configs['iter'] = args.iter

            use_data_dict = {}
            d_index = aim_date.index(d)
            if d_index <= args.date_size:
                start_d = aim_date[0]
            else:
                start_d = aim_date[d_index - args.date_size]

            configs["start_date"] = start_d
            print("start date is: %s" % start_d)

            true_base_data = true_data[((true_data.ds >= start_d) &
                                        (true_data.ds < d))].reset_index(drop=True)

            true_base_data_unit = true_base_data.groupby("unit").agg({"total_qps":"mean"}).reset_index()
            true_base_data_unit = true_base_data_unit.sort_values("total_qps",ascending=False).reset_index(drop=True)
            level_step = true_base_data_unit.shape[0]//args.qps_level_num

            true_base_data_unit["levels"] = pd.Series([ik for ik in range(true_base_data_unit.shape[0])])
            true_base_data_unit["levels"] = true_base_data_unit["levels"]//level_step

            print(true_base_data_unit.head(24))



            for j in aim_models.keys():
                use_data_dict[j] = app_heter_data[j][(app_heter_data[j]['ds'] >= start_d)
                                                     &(app_heter_data[j]['ds'] < d)]
                use_data_dict[j] = pd.merge(use_data_dict[j],true_base_data_unit[["unit","levels"]],on="unit",how="left")
                use_data_dict[j] = use_data_dict[j].sort_values(args.sample_time).reset_index(drop=True)

                print(j)
                print(use_data_dict[j].shape)
                print(use_data_dict[j].dropna().shape)

                print(use_data_dict[j].columns)

            date_path = os.path.join(app_out_res_path, d)
            if not os.path.exists(date_path):
                os.makedirs(date_path, exist_ok=True)

            save_config(config=configs, filename=os.path.join(date_path, "exp_config.json"))
            norm = False
            if args.norm >= 1:
                norm = True
            else:
                norm = False

            standard = True
            if args.standard < 1:
                standard = False
            else:
                standard = True

            n_component = None
            if args.n_components > 0:
                n_component = args.n_components

            # total_corr_pdf_list, aim_corr_pdf_list, aim_times, bic_aic_results_list = find_train_range_gmm(x_df_dict=use_data_dict,
            #                                 base_model=args.base_model, init_size=args.init_size*args.chunck_size, freq=args.freq, step=args.step*args.chunck_size,
            #                                 n_components=n_component,
            #                                  select_n=args.select_n, diff_limit=args.diff_limit,
            #                                  diff_win=args.diff_win, unit=args.unit, x_col="total_qps", y_col='total_usage', test_size=args.test_size,
            #                                  sample_time=args.sample_time, norm=norm, x_unit=args.x_unit, y_unit=args.y_unit,
            #                                 block_size=args.chunck_size*args.block_size,
            #                                  standard=standard, lateset_range=2, mode=args.gmm_mode, norm_mode=args.norm_mode, cov_col=args.cov_col,
            #                                  pdc_col=args.pdc_col)

            device = torch.device('cuda:%d' % int(args.gpu) if torch.cuda.is_available() and use_gpu else 'cpu')
            for ir in range(args.iter):
                if args.total_mode > 0:
                    total_corr_pdf_list, aim_corr_pdf_list, aim_times = find_train_range_dist(x_df_dict=use_data_dict, base_model=args.base_model,
                                          init_size=args.init_size*args.chunck_size, freq=args.freq, step=args.step*args.chunck_size,
                                          diff_win=args.diff_win, unit=args.unit, x_col="total_qps", y_col='total_usage',
                                          model=args.drift_model,
                                          p_val=args.p_val,
                                          if_update_ref=if_update_ref,
                                          x_ref_preprocessed=x_ref_preprocessed,
                                          preprocess_at_init=preprocess_at_init,
                                          inverse=inverse, n_permutations=args.n_permutations, lambda_rd_max=args.lambda_rd_max,
                                          update_x_ref=args.update_x_ref, update_ref=args.update_ref,
                                          backend=args.backend, prop_c_held=args.prop_c_held, n_folds=args.n_folds,
                                          device=device, early_stop=early_stop,
                                          context_col=context_col,
                                          heter_context=heter_context,
                                          replicas_col=args.replicas_col,
                                          heter_method=args.heter_method,
                                          sample_time=args.sample_time, norm=norm, x_unit=args.x_unit, y_unit=args.y_unit,
                                          block_size=args.chunck_size*args.block_size,
                                          standard=standard, lateset_range=2, mode=args.gmm_mode, norm_mode=args.norm_mode, cov_col=args.cov_col,
                                          pdc_col=args.pdc_col, save=save_model, save_path=date_path,use_gpu=use_gpu)
                else:
                    total_corr_pdf_list, aim_corr_pdf_list, aim_times = find_train_range_dist(x_df_dict=use_data_dict, base_model=args.base_model,
                                          init_size=args.init_size * args.chunck_size, freq=args.freq,
                                          step=args.step * args.chunck_size,
                                          diff_win=args.diff_win, unit=args.unit, x_col="qps__Q_s", y_col='cpu_usage',
                                          model=args.drift_model,
                                          p_val=args.p_val,
                                          if_update_ref=if_update_ref,
                                          x_ref_preprocessed=x_ref_preprocessed,
                                          preprocess_at_init=preprocess_at_init,
                                          inverse=inverse, n_permutations=args.n_permutations,
                                          lambda_rd_max=args.lambda_rd_max,
                                          update_x_ref=args.update_x_ref, update_ref=args.update_ref,
                                          backend=args.backend, prop_c_held=args.prop_c_held, n_folds=args.n_folds,
                                          device=device, early_stop=early_stop,
                                          context_col=context_col,
                                          heter_context=heter_context,
                                          replicas_col=args.replicas_col,
                                          heter_method=args.heter_method,
                                          sample_time=args.sample_time, norm=norm, x_unit=args.x_unit,
                                          y_unit=args.y_unit,
                                          block_size=args.chunck_size * args.block_size,
                                          standard=standard, lateset_range=2, mode=args.gmm_mode,
                                          norm_mode=args.norm_mode, cov_col=args.cov_col,
                                          pdc_col=args.pdc_col, save=save_model, save_path=date_path,use_gpu=use_gpu)

                for jj in total_corr_pdf_list.keys():
                    total_corr_pdf_list[jj].to_csv(os.path.join(date_path,"total_cor_%s_%d.csv" % (jj,ir)),index=False)

                for jj in aim_corr_pdf_list.keys():
                    aim_corr_pdf_list[jj].to_csv(os.path.join(date_path,"aim_cor_%s_%d.csv" % (jj,ir)),index=False)

                for jj in aim_times.keys():
                    aim_times[jj].to_csv(os.path.join(date_path,"aim_time_%s_%d.csv" % (jj,ir)),index=False)














