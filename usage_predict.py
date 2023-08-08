from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR  # 随机森林模块
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR  # 线性回归模块
from utils import MAPE,preprocess_qps_usage_data,generate_bayes_label_for_train,compute_bayesian_posterior
from skgrf.ensemble import GRFForestLocalLinearRegressor

import numpy as np
import pandas as pd


class PredictUsageModel():
    def __init__(self, bayies_x_metric='rt', bayies_y_metric='cpu_uti', reg_x_metric='qps__Q_s',
                 reg_y_metric='cpu_usage', x_split=20, y_split=20, test_size=0.2, norm=False,
                 lr_params={}, rfr_params={}, xgbr_params={},
                 gbdtr_params={},grf_params={}, perf_metric='r2score'):
        '''
        bayies_x_metric:进行贝叶斯推断的X维度(默认为rt)
        bayies_y_metric:进行贝叶斯推断的Y维度(默认为cpu utilization)
        reg_x_metric: 进行回归预测的输入维度
        reg_y_metric:进行回归预测的输出维度
        x_split:贝叶斯推断时对X维度划分多少级level
        y_split：贝叶斯推断时对Y维度划分多少级level
        lr_params：线性回归器指定参数
        rfr_params：随机森林回归器指定参数
        xgbr_params：XGBBoost回归器指定参数
        gbdtr_params：GBDT回归器指定参数
        '''

        self.bayies_x_metric = bayies_x_metric
        self.bayies_y_metric = bayies_y_metric
        self.bayies_x_label = "%s_label" % bayies_x_metric
        self.bayies_y_label = "%s_label" % bayies_y_metric
        self.reg_x_metric = reg_x_metric
        self.reg_y_metric = reg_y_metric
        self.test_size = test_size
        self.norm = norm
        self.x_split = x_split
        self.y_split = y_split
        #         self.max_x_split = max_x_split
        #         self.max_y_split = max_y_split
        self.x_stat = {}
        self.y_stat = {}
        self.perf_metric = perf_metric
        self.reg_params = {'lr': {}, 'gbdtr': {}, 'xgbr': {}, 'rfr': {},"grf":{}}
        self.reg_performance = {'lr': {}, 'gbdtr': {}, 'xgbr': {}, 'rfr': {},"grf":{}}
        self.mean_stds = {}
        self.reg_trained = {'lr': False, 'gbdtr': False, 'xgbr': False, 'rfr': False,'grf': False}
        self.reg_models = {'lr': LR(),
                           'xgbr': XGBR(n_estimators=500, learning_rate=0.25, base_score=0.8, max_depth=6, gamma=0.3),
                           'gbdtr': GradientBoostingRegressor(n_estimators=400, learning_rate=0.2, loss="huber",
                                                              alpha=0.9, min_samples_split=5, min_samples_leaf=3),
                           'rfr': RFR(n_estimators=300),
                           "grf": GRFForestLocalLinearRegressor(ll_n_estimators=100, ll_split_weight_penalty=False, ll_split_lambda=0.1, ll_split_variables=None, ll_split_cutoff=None, ll_equalize_cluster_weights=False,
                ll_sample_fraction=0.5, ll_mtry=None, ll_min_node_size=5, ll_honesty=True,
                ll_honesty_fraction=0.5, ll_honesty_prune_leaves=True, ll_alpha=0.05, ll_imbalance_penalty=0, ll_ci_group_size=2,
                ll_n_jobs=-1, ll_seed=42, ll_enable_tree_details=False,qu_n_estimators=100,
                 samples=25, regression_splitting=False,
                 qu_equalize_cluster_weights=False, qu_sample_fraction=0.5, qu_mtry=None, qu_min_node_size=5,
                 qu_honesty=True, qu_honesty_fraction=0.5, qu_honesty_prune_leaves=True,
                 qu_alpha=0.05, qu_imbalance_penalty=0, qu_n_jobs=- 1, qu_seed=42, qu_enable_tree_details=False)}

        if lr_params:
            self.reg_models['lr'] = LR(**lr_params)
            self.reg_params['lr'] = lr_params

        if rfr_params:
            self.reg_models['rfr'] = RFR(**rfr_params)
            self.reg_params['rfr'] = rfr_params

        if xgbr_params:
            self.reg_models['xgbr'] = XGBR(**xgbr_params)
            self.reg_params['xgbr'] = xgbr_params

        if gbdtr_params:
            self.reg_models['gbdtr'] = GradientBoostingRegressor(**gbdtr_params)
            self.reg_params['gbdtr'] = gbdtr_params

        if grf_params:
            self.reg_models['grf'] = GRFForestLocalLinearRegressor(**grf_params)
            self.reg_params['grf'] = grf_params

        if isinstance(reg_x_metric, list):
            for k in reg_x_metric:
                self.mean_stds[k] = {'mean': 0.0, 'std': 1.0}
        else:
            self.mean_stds[reg_x_metric] = {'mean': 0.0, 'std': 1.0}

        if isinstance(reg_y_metric, list):
            for k in reg_y_metric:
                self.mean_stds[k] = {'mean': 0.0, 'std': 1.0}
        else:
            self.mean_stds[reg_y_metric] = {'mean': 0.0, 'std': 1.0}

        for j in range(x_split):
            self.x_stat['%s_p%d' % (bayies_x_metric, int(j / x_split * 100))] = 0.0

        for j in range(y_split):
            self.y_stat['%s_p%d' % (bayies_y_metric, int(j / y_split * 100))] = 0.0

        self.bayies_posterior_map = {}
        for i in range(x_split):
            self.bayies_posterior_map[i] = {}
            for j in range(y_split):
                self.bayies_posterior_map[i][j] = 0.0
        self.bayies_posterior_matrix = np.zeros([x_split, y_split])

    def set_parameters_of_reg(self, reg='lr', params={}):
        self.reg_params[reg] = params
        if reg == 'lr':
            self.reg_models[reg] = LR(**params)
        if reg == 'gbdtr':
            self.reg_models[reg] = GradientBoostingRegressor(**params)
        if reg == 'xgbr':
            self.reg_models[reg] = XGBR(**params)
        if reg == 'rfr':
            self.reg_models[reg] = RFR(**params)
        if reg == 'rfr':
            self.reg_models[reg] = GRFForestLocalLinearRegressor(**params)

    def generate_bayies_labels(self, x_pdf):
        outdf, self.x_stat, self.y_stat = generate_bayes_label_for_train(x_pdf, x_metric=self.bayies_x_metric,
                                                                         y_metric=self.bayies_y_metric,
                                                                         x_split=self.x_split, y_split=self.y_split)
        return outdf

    def train_bayies_posterior(self, x_pdf, x_label=None, y_label=None):
        if x_label == None or x_label == "":
            x_label_metric = self.bayies_x_label
        else:
            x_label_metric = x_label

        if y_label == None or y_label == "":
            y_label_metric = self.bayies_y_label
        else:
            y_label_metric = y_label

        self.bayies_posterior_map, self.bayies_posterior_matrix = compute_bayesian_posterior(x_label_metric,
                                                                                             y_label_metric, x_pdf)

    def train_bayies_process(self, x_pdf, bayies_x_metric=None, bayies_y_metric=None):
        if bayies_x_metric != None and bayies_x_metric != "":
            self.bayies_x_metric = bayies_x_metric
            self.bayies_x_label = "%s_label" % bayies_x_metric

        x_label_metric = self.bayies_x_label

        if bayies_x_metric != None and bayies_x_metric != "":
            self.bayies_y_metric = bayies_y_metric
            self.bayies_y_label = "%s_label" % bayies_y_metric

        y_label_metric = self.bayies_y_label

        out_df = self.generate_bayies_labels(x_pdf)

        self.train_bayies_posterior(x_pdf, x_label=x_label_metric, y_label=y_label_metric)

        return out_df

    def train_specific_regressor_qps_to_usage(self, x_pdf, test_size=-1, reg='grf', reg_params={}):
        #         preprocess_qps_usage_data(x_pdf, test_size=0.2, x_column='total_qps', y_column='total_usage',norm=False)
        if test_size < 0 or test_size >= 1:
            X_train, X_test, y_train, y_test, self.mean_stds = preprocess_qps_usage_data(x_pdf,
                                                                                         test_size=self.test_size,
                                                                                         x_column=self.reg_x_metric,
                                                                                         y_column=self.reg_y_metric,
                                                                                         norm=self.norm)
        else:
            X_train, X_test, y_train, y_test, self.mean_stds = preprocess_qps_usage_data(x_pdf, test_size=test_size,
                                                                                         x_column=self.reg_x_metric,
                                                                                         y_column=self.reg_y_metric,
                                                                                         norm=self.norm)
        if reg_params:
            self.set_parameters_of_reg(reg=reg, params=reg_params)
        self.reg_models[reg].fit(X_train, y_train)
        test_res = {}
        self.reg_performance[reg]['r2score'] = self.reg_models[reg].score(X_test, y_test)
        self.reg_performance[reg]['mse'] = MSE(y_true=y_test, y_pred=self.reg_models[reg].predict(X_test))
        self.reg_performance[reg]['mape'] = MAPE(y_true=y_test, y_pred=self.reg_models[reg].predict(X_test))
        test_res[self.reg_performance[reg][self.perf_metric]] = reg
        print("%s: r2score: %f; mse: %f mape: %f" % (
        reg, self.reg_performance[reg]['r2score'], self.reg_performance[reg]['mse'], self.reg_performance[reg]['mape']))
        for reg_key in self.reg_trained.keys():
            if self.reg_trained[reg_key]:
                test_res[self.reg_performance[reg_key][self.perf_metric]] = reg_key

        test_metric_values = list(test_res.keys())
        test_metric_values.sort()
        if self.perf_metric == 'r2score':
            self.reg_models['best'] = test_res[test_metric_values[-1]]
        else:
            self.reg_models['best'] = test_res[test_metric_values[0]]

    def train_regressor_qps_to_usage(self, x_pdf, test_size=-1, lr_params={}, xgbr_params={}, gbdtr_params={},
                                     rfr_params={},grf_params={}):
        if test_size < 0 or test_size >= 1:
            X_train, X_test, y_train, y_test, self.mean_stds = preprocess_qps_usage_data(x_pdf, test_size=self.test_size,
                                                                                         x_column=self.reg_x_metric,
                                                                                         y_column=self.reg_y_metric,
                                                                                         norm=self.norm)
        else:
            X_train, X_test, y_train, y_test, self.mean_stds = preprocess_qps_usage_data(x_pdf, test_size=test_size,
                                                                                         x_column=self.reg_x_metric,
                                                                                         y_column=self.reg_y_metric,
                                                                                         norm=self.norm)
        if lr_params:
            self.set_parameters_of_reg(reg='lr', params=lr_params)
        if xgbr_params:
            self.set_parameters_of_reg(reg='xgbr', params=xgbr_params)
        if gbdtr_params:
            self.set_parameters_of_reg(reg='gbtr', params=gbdtr_params)
        if rfr_params:
            self.set_parameters_of_reg(reg='rfr', params=rfr_params)
        if grf_params:
            self.set_parameters_of_reg(reg='grf', params=grf_params)

        test_res = {}
        for k in self.reg_models.keys():
            self.reg_models[k].fit(X_train, y_train)
            self.reg_performance[k]['r2score'] = self.reg_models[k].score(X_test, y_test)
            self.reg_performance[k]['mse'] = MSE(y_true=y_test, y_pred=self.reg_models[k].predict(X_test))
            self.reg_performance[k]['mape'] = MAPE(y_true=y_test, y_pred=self.reg_models[k].predict(X_test))

            test_res[self.reg_performance[k][self.perf_metric]] = k
            print("%s: r2score: %f; mse: %f mape: %f" % (
            k, self.reg_performance[k]['r2score'], self.reg_performance[k]['mse'], self.reg_performance[k]['mape']))

        test_metric_values = list(test_res.keys())
        test_metric_values.sort()
        if self.perf_metric == 'r2score':
            self.reg_models['best'] = test_res[test_metric_values[-1]]
        else:
            self.reg_models['best'] = test_res[test_metric_values[0]]

    def train(self, x_pdf, stage='total', bayies_x_metric=None, bayies_y_metric=None, test_size=-1, reg=None,
              lr_params={}, xgbr_params={},
              gbdtr_params={}, rfr_params={}, grf_params={},
              reg_params={}):
        if stage == 'total':
            out_df = self.train_bayies_process(x_pdf=x_pdf, bayies_x_metric=bayies_x_metric,
                                               bayies_y_metric=bayies_y_metric)
            if reg == None or reg == '':
                self.train_regressor_qps_to_usage(out_df, test_size=test_size, lr_params=lr_params,
                                                  xgbr_params=xgbr_params, gbdtr_params=gbdtr_params,
                                                  rfr_params=rfr_params,grf_params=grf_params)
                for reg_key in self.reg_trained.keys():
                    self.reg_trained[reg_key] = True
            else:
                if reg not in self.reg_models.keys():
                    reg = 'lr'
                self.train_specific_regressor_qps_to_usage(out_df, test_size, reg=reg, reg_params=reg_params)
                self.reg_trained[reg] = True

        elif stage == 'bayies':
            out_df = self.train_bayies_process(x_pdf=x_pdf, bayies_x_metric=bayies_x_metric,
                                               bayies_y_metric=bayies_y_metric)
        elif stage == 'regressor':
            out_df = x_pdf.copy()
            if reg == None or reg == '':
                self.train_regressor_qps_to_usage(out_df, test_size=test_size, lr_params=lr_params,
                                                  xgbr_params=xgbr_params, gbdtr_params=gbdtr_params,
                                                  rfr_params=rfr_params,grf_params=grf_params)

                for reg_key in self.reg_trained.keys():
                    self.reg_trained[reg_key] = True
            else:
                if reg not in self.reg_models.keys():
                    reg = 'grf'

                self.train_specific_regressor_qps_to_usage(out_df, test_size=test_size, reg=reg, reg_params=reg_params)
                self.reg_trained[reg] = True
        else:
            out_df = self.train_bayies_process(x_pdf=x_pdf, bayies_x_metric=bayies_x_metric,
                                               bayies_y_metric=bayies_y_metric)
            if reg == None or reg == '':
                self.train_regressor_qps_to_usage(out_df, test_size=test_size, lr_params=lr_params,
                                                  xgbr_params=xgbr_params, gbdtr_params=gbdtr_params,
                                                  rfr_params=rfr_params,grf_params=grf_params)
            else:
                self.train_specific_regressor_qps_to_usage(out_df, test_size, reg=reg, reg_params=reg_params)
        return out_df

    def predict_usage(self, future_qps, reg='lr', cpu_uti_limit=0.4, rt_mean_limit=None, violate=0.1):
        '''
        参数：
        future_qps：为未来的QPS总量,或者输入值的字典（dict）
        cpu_uti_limit: (0,1]为限制的CPU利用率峰值
        reg：为预测当前总QPS容量时选择的regressor，目前支持：lr,xreg,xgbr,gbdtr四种
        rt_mean_limit： 用户指定的期望的RT平均值上限
        violate：当使用贝叶斯推断指定rt_mean上限对应的CPU阈值时，设定置信度/允许的QoS违反的概率
        返回值：
        cpu_quota_predict: 满足QPS用量的CPU配额。
        '''
        if reg not in self.reg_models.keys():
            reg = 'lr'

        if reg == 'best':
            reg = self.reg_models['best']

        if isinstance(self.reg_x_metric, list):
            if isinstance(future_qps, dict):
                input_data = []
                for k in self.reg_x_metric:
                    input_dim_data = (future_qps[k] - self.mean_stds[k]['mean']) / self.mean_stds[k]['std']
                    input_data.append(input_dim_data)
                input_data = np.asarray(input_data).transpose()
                intput_data = input_data.reshape(-1, len(self.reg_x_metric))
            else:
                print("please input the data as {'column1':[values1],'column2':['values2']...}")
                return 0
        else:
            #              - self.mean_stds[self.reg_x_metric]['mean'] /self.mean_stds[self.reg_x_metric]['std']
            if isinstance(future_qps,float):
                if self.norm:
                    input_data = np.asarray([(future_qps - self.mean_stds[self.reg_x_metric]['mean']) /
                                             self.mean_stds[self.reg_x_metric]['std']]).reshape(-1, 1)
                else:
                    input_data = np.asarray([future_qps]).reshape(-1, 1)
            else:
                if self.norm:
                    input_data = np.asarray((future_qps - self.mean_stds[self.reg_x_metric]['mean']) /
                                             self.mean_stds[self.reg_x_metric]['std']).reshape(-1, 1)
                else:
                    input_data = np.asarray(future_qps).reshape(-1, 1)


        predict_cpu_usage = np.array(self.reg_models[reg].predict(input_data)).squeeze()
        if self.norm:
            predict_cpu_usage = predict_cpu_usage * self.mean_stds[self.reg_y_metric]['std'] + \
                                self.mean_stds[self.reg_y_metric]['mean']
        else:
            #          * self.mean_stds[self.reg_y_metric]['std'] + self.mean_stds[self.reg_y_metric]['std']
            predict_cpu_usage = predict_cpu_usage
        # print("predict usage: ")
        # print(predict_cpu_usage)
        return predict_cpu_usage