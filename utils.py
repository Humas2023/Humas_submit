import numpy as np

# from sklearn.mixture import GaussianMixture as GMM
import datetime
# from xgboost import XGBRegressor as XGBR
# from sklearn.ensemble import RandomForestRegressor as RFR  # 随机森林模块
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn import datasets
from sklearn.model_selection import KFold,cross_val_score,train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.linear_model import LinearRegression as LR  # 线性回归模块
import pandas as pd


def generate_bayes_label_for_train(x_pdf, x_metric='rt', y_metric='cpu_uti__pct', qps_col='qps__Q_s', x_split=20,
                                   y_split=20):
    cpu_stat = {}
    rt_stat = {}
    out_df = x_pdf[x_pdf[qps_col] > 0].reset_index(drop=True)
    out_df, rt_stat, cpu_stat = compute_bayes_label(x_pdf, x_split=x_split, y_split=y_split, x_metric=x_metric,
                                                    y_metric=y_metric)
    return out_df, rt_stat, cpu_stat


# 对该应用数据进行贝叶斯推断，推断不同RT级别与CPU级别情况下的条件概率：
def compute_bayes_label(x_pdf, x_split=20, y_split=20, x_metric='rt', y_metric="cpu_uti"):
    x_inter = 1 / x_split
    y_inter = 1 / y_split

    x_label_split = []
    y_label_split = []

    x_label = []
    y_label = []

    for i in range(x_split):
        x_label_split.append(x_pdf[x_metric].dropna().quantile(x_inter * i))

    for i in range(y_split):
        y_label_split.append(x_pdf[y_metric].dropna().quantile(y_inter * i))

    #     print(len(x_pdf.index))
    #     print(len(x_raw))
    for i in x_pdf.index:
        #         print(x_pdf[y_metric][i])
        if x_pdf[y_metric][i] < y_label_split[1]:
            y_label.append(0)
        elif x_pdf[y_metric][i] >= y_label_split[y_split - 1]:
            y_label.append(y_split - 1)
        else:
            for j in range(len(y_label_split) - 1, 0, -1):
                if x_pdf[y_metric][i] >= y_label_split[j]:
                    y_label.append(j)
                    break

        if x_pdf[x_metric][i] < x_label_split[1]:
            x_label.append(0)
        elif x_pdf[x_metric][i] >= x_label_split[x_split - 1]:
            x_label.append(x_split - 1)
        else:
            for j in range(len(x_label_split) - 1, 0, -1):
                if x_pdf[x_metric][i] >= x_label_split[j]:
                    x_label.append(j)
                    break

    #     print(len(y_label))
    #     print(len(x_label))
    x_pdf['%s_label' % y_metric] = pd.Series(y_label)
    x_pdf['%s_label' % x_metric] = pd.Series(x_label)

    #     print(len(x_label))

    x_statics = {}
    y_statics = {}

    for i in range(x_split):
        x_statics["%s_p%d" % (x_metric, int(i * x_inter * 100))] = [x_label_split[i]]
    x_statics['%s_p99' % x_metric] = [x_pdf[x_metric].max()]
    for i in range(y_split):
        y_statics["%s_p%d" % (y_metric, int(i * x_inter * 100))] = [y_label_split[i]]
    y_statics['%s_p99' % y_metric] = [x_pdf[y_metric].max()]

    return x_pdf, pd.DataFrame(x_statics), pd.DataFrame(y_statics)


def compute_bayesian_posterior(x_label_metric, y_label_metric, x_pdf):
    x_labels = x_pdf[x_label_metric].unique().tolist()
    x_labels.sort()
    y_labels = x_pdf[y_label_metric].unique().tolist()
    y_labels.sort()
    result_map = {}  # 即result[i][j] = P(X=i|Y=j)，即CPU利用率不同level条件下,RT处于每一level的概率。
    results = []
    total_len = x_pdf.shape[0]
    for i in range(len(x_labels)):
        x_res = x_pdf[x_pdf[x_label_metric] == x_labels[i]]
        if x_labels[i] not in result_map.keys():
            result_map[int(x_labels[i])] = {}
            single_results = []
        for j in range(len(y_labels)):
            y_res = x_pdf[x_pdf[y_label_metric] == y_labels[j]]
            p_y = y_res.shape[0] / total_len
            xy_res = x_res[x_res[y_label_metric] == y_labels[j]]
            p_xy = xy_res.shape[0] / total_len
            result_map[x_labels[i]][y_labels[j]] = p_xy / p_y
            single_results.append(result_map[x_labels[i]][y_labels[j]])

        results.append(single_results)

    return result_map, np.asarray(results)

def out_date(year,day):
    fir_day = datetime.datetime(year,1,1)
    zone = datetime.timedelta(days=day-1)
    return datetime.datetime.strftime(fir_day + zone, "%Y-%m-%d")

def out_ds(year,day):
    fir_day = datetime.datetime(year,1,1)
    zone = datetime.timedelta(days=day-1)
    return datetime.datetime.strftime(fir_day + zone, "%Y%m%d")

def which_day(year, month, date):
    end = datetime.date(year, month, date)
    start = datetime.date(year, 1, 1)
    return (end - start).days + 1

def get_selected_ds(x_pdf,start_col='start',end_col='end',additional=0):
    day_sets = []
    for i in range(x_pdf.shape[0]):
        tmp_start = x_pdf[start_col].values[i].split()[0]
        tmp_end = x_pdf[end_col].values[i].split()[0]

        tmp_starts = tmp_start.split("-")
        tmp_ends = tmp_end.split("-")

        tmp_start_day = which_day(year=int(tmp_starts[0]),month=int(tmp_starts[1]),date=int(tmp_starts[2]))
        tmp_end_day = which_day(year=int(tmp_ends[0]),month=int(tmp_ends[1]),date=int(tmp_ends[2]))

        for k in range(tmp_start_day-additional,tmp_end_day+1):
            day_sets.append(out_ds(year=int(tmp_starts[0]),day=k))

    day_sets = list(set(day_sets))
    return day_sets


def preprocess_qps_usage_data(x_pdf, test_size=0.2, x_column='total_qps', y_column='total_usage',norm=False):
   
    mean_stds = {}
    if isinstance(x_column, list):
        for k in x_column:
            mean_stds[k] = {}
            mean_stds[k]['mean'] = x_pdf[k].mean()
            mean_stds[k]['std'] = x_pdf[k].std()
    else:
        mean_stds[x_column] = {}
        mean_stds[x_column]['mean'] = x_pdf[x_column].mean()
        mean_stds[x_column]['std'] = x_pdf[x_column].std()

    if isinstance(y_column, list):
        for k in y_column:
            mean_stds[k] = {}
            mean_stds[k]['mean'] = x_pdf[k].mean()
            mean_stds[k]['std'] = x_pdf[k].std()
    else:
        mean_stds[y_column] = {}
        mean_stds[y_column]['mean'] = x_pdf[y_column].mean()
        mean_stds[y_column]['std'] = x_pdf[y_column].std()

    if isinstance(x_column, list):
        X_data = x_pdf[x_column].copy()
        for j in x_column:
            if norm:
                X_data[j] = ((X_data[j])-X_data[j].mean()) / (X_data[j].std())

        X_data = X_data[x_column].to_numpy()
    else:
        #         -x_pdf[[x_column]].mean() /(x_pdf[[x_column]].std())
        X_data = (x_pdf[[x_column]]).copy()
        if norm:
            X_data[x_column] = (x_pdf[x_column]-x_pdf[x_column].mean()) / (x_pdf[x_column].std())

        X_data = ((x_pdf[[x_column]])).to_numpy()

    if isinstance(y_column, list):
        Y_data = x_pdf[y_column].copy()
        for j in y_column:
            #             -Y_data[[j]].mean() /(Y_data[[j]].std())
            if norm:
                Y_data[j] = ((Y_data[j])-Y_data[j].mean()) / (Y_data[j].std())

        Y_data = Y_data[y_column].to_numpy()

    else:
        #         -x_pdf[[y_column]].mean() /(x_pdf[[y_column]].std())
        Y_data = ((x_pdf[[y_column]])).copy()
        if norm:
            Y_data[y_column] = ((x_pdf[y_column]) - x_pdf[y_column].mean()) / (x_pdf[y_column].std())

        Y_data = Y_data[[y_column]].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=420)

    return X_train, X_test, y_train, y_test, mean_stds

def MAPE(y_true,y_pred):
    # print(type(y_true))
    # print(type(y_pred))
    using_df = pd.DataFrame()
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    using_df['y'] = pd.Series(y_true.tolist())
    using_df = using_df.reset_index(drop=True)
    using_df['yhat'] = pd.Series(y_pred.tolist())
    using_df = using_df.reset_index(drop=True)
    # print(using_df)
    using_df['mape'] = ((using_df['y']-using_df['yhat'])/using_df['y']).abs()
    return using_df['mape'].replace([np.inf, -np.inf], np.nan).mean()