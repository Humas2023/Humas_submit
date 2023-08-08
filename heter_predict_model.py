import numpy as np
import pandas as pd
import joblib
from skgrf.ensemble import GRFForestLocalLinearRegressor
from skgrf.ensemble import GRFForestQuantileRegressor
class UsageMapPredictor():
    def __init__(self,ll_n_estimators=100, ll_split_weight_penalty=False, ll_split_lambda=0.1, ll_split_variables=None, ll_split_cutoff=None, ll_equalize_cluster_weights=False,
                ll_sample_fraction=0.5, ll_mtry=None, ll_min_node_size=5, ll_honesty=True,
                ll_honesty_fraction=0.5, ll_honesty_prune_leaves=True, ll_alpha=0.05, ll_imbalance_penalty=0, ll_ci_group_size=2,
                ll_n_jobs=-1, ll_seed=42, ll_enable_tree_details=False,qu_n_estimators=100,
                 samples=25, regression_splitting=False,
                 qu_equalize_cluster_weights=False, qu_sample_fraction=0.5, qu_mtry=None, qu_min_node_size=5,
                 qu_honesty=True, qu_honesty_fraction=0.5, qu_honesty_prune_leaves=True,
                 qu_alpha=0.05, qu_imbalance_penalty=0, qu_n_jobs=- 1, qu_seed=42, qu_enable_tree_details=False):



        quantiles = [i/samples for i in range(1+samples)]

        quantiles[0] = 0+1/(samples*10)
        quantiles[-1] = 1-1/(samples*10)

        self.ll_n_estimators = ll_n_estimators
        self.samples = samples
        self.ll_split_weight_penalty = ll_split_weight_penalty
        self.ll_split_lambda = ll_split_lambda
        self.ll_split_variables = ll_split_variables
        self.ll_split_cutoff = ll_split_cutoff
        self.ll_equalize_cluster_weights = ll_equalize_cluster_weights,
        self.ll_sample_fraction = ll_sample_fraction
        self.ll_mtry = ll_mtry
        self.ll_min_node_size = ll_min_node_size
        self.ll_honesty = ll_honesty
        self.ll_honesty_fraction = ll_honesty_fraction
        self.ll_honesty_prune_leaves = ll_honesty_prune_leaves
        self.ll_alpha = ll_alpha
        self.ll_imbalance_penalty = ll_imbalance_penalty
        self.ll_ci_group_size = ll_ci_group_size
        self.ll_n_jobs = ll_n_jobs
        self.ll_seed = ll_seed
        self.ll_enable_tree_details = ll_enable_tree_details

        self.qu_n_estimators = qu_n_estimators
        self.quantiles = quantiles
        self.regression_splitting = regression_splitting
        self.qu_equalize_cluster_weights = qu_equalize_cluster_weights
        self.qu_sample_fraction = qu_sample_fraction
        self.qu_mtry = qu_mtry
        self.qu_min_node_size = qu_min_node_size
        self.qu_honesty = qu_honesty
        self.qu_honesty_fraction = qu_honesty_fraction
        self.qu_honesty_prune_leaves = qu_honesty_prune_leaves
        self.qu_alpha = qu_alpha
        self.qu_imbalance_penalty = qu_imbalance_penalty
        self.qu_n_jobs = qu_n_jobs
        self.qu_seed = qu_seed
        self.qu_enable_tree_details = qu_enable_tree_details

        self.ll_forest = GRFForestLocalLinearRegressor(n_estimators=self.ll_n_estimators,
                                                       ll_split_weight_penalty=self.ll_split_weight_penalty,
                                                       ll_split_lambda=self.ll_split_lambda,
                                                       ll_split_variables=self.ll_split_variables,
                                                       ll_split_cutoff=self.ll_split_cutoff,
                                                       equalize_cluster_weights=self.ll_equalize_cluster_weights,
                                                       sample_fraction=self.ll_sample_fraction, mtry=self.ll_mtry,
                                                       min_node_size=self.ll_min_node_size, honesty=self.ll_honesty,
                                                       honesty_fraction=self.ll_honesty_fraction,
                                                       honesty_prune_leaves=self.ll_honesty_prune_leaves,
                                                       alpha=self.ll_alpha, imbalance_penalty=self.ll_imbalance_penalty,
                                                       ci_group_size=self.ll_ci_group_size,
                                                       n_jobs=self.ll_n_jobs, seed=self.ll_seed,
                                                       enable_tree_details=self.ll_enable_tree_details)

        self.qu_forest = GRFForestQuantileRegressor(n_estimators=self.qu_n_estimators, quantiles=self.quantiles,
                                                    regression_splitting=self.regression_splitting,
                                                    equalize_cluster_weights=self.qu_equalize_cluster_weights,
                                                    sample_fraction=self.qu_sample_fraction, mtry=self.qu_mtry,
                                                    min_node_size=self.qu_min_node_size,
                                                    honesty=self.qu_honesty, honesty_fraction=self.qu_honesty_fraction,
                                                    honesty_prune_leaves=self.qu_honesty_prune_leaves, alpha=self.qu_alpha,
                                                    imbalance_penalty=self.qu_imbalance_penalty,
                                                    n_jobs=self.qu_n_jobs, seed=self.qu_seed,
                                                    enable_tree_details=self.qu_enable_tree_details)

    def set_ll_n_estimators(self,ll_n_estimators):
        self.ll_n_estimators = ll_n_estimators

    def load_ll_forest(self,file):
        # estimator = joblib.load('逻辑回归.pkl')
        self.ll_forest = joblib.load(file)

    def load_qu_forest(self,file):
        self.qu_forest = joblib.load(file)

    def set_ll_split_weight_penalty(self,ll_split_weight_penalty):
        self.ll_split_weight_penalty = ll_split_weight_penalty

    def set_ll_split_lambda(self,ll_split_lambda):
        self.ll_split_lambda = ll_split_lambda

    def set_ll_split_variables(self,ll_split_variables):
        self.ll_split_variables = ll_split_variables

    def set_ll_split_cutoff(self,ll_split_cutoff):
        self.ll_split_cutoff = ll_split_cutoff

    def set_ll_equalize_cluster_weights(self,ll_equalize_cluster_weights):
        self.ll_equalize_cluster_weights = ll_equalize_cluster_weights

    def set_ll_sample_fraction(self,ll_sample_fraction):
        self.ll_sample_fraction = ll_sample_fraction

    def set_ll_mtry(self,ll_mtry):
        self.ll_mtry = ll_mtry

    def set_ll_min_node_size(self,ll_min_node_size):
        self.ll_min_node_size = ll_min_node_size

    def se_ll_honesty(self,ll_honesty):
        self.ll_honesty = ll_honesty

    def set_ll_honesty_fraction(self,ll_honesty_fraction):
        self.ll_honesty_fraction = ll_honesty_fraction

    def set_ll_honesty_prune_leaves(self,ll_honesty_prune_leaves):
        self.ll_honesty_prune_leaves = ll_honesty_prune_leaves

    def set_ll_alpha(self,ll_alpha):
        self.ll_alpha = ll_alpha

    def set_ll_imbalance_penalty(self,ll_imbalance_penalty):
        self.ll_imbalance_penalty = ll_imbalance_penalty

    def set_ll_ci_group_size(self,ll_ci_group_size):
        self.ll_ci_group_size = ll_ci_group_size

    def set_ll_n_jobs(self,ll_n_jobs):
        self.ll_n_jobs = ll_n_jobs

    def set_ll_seed(self,ll_seed):
        self.ll_seed = ll_seed

    def set_ll_enable_tree_details(self,ll_enable_tree_details):
        self.ll_enable_tree_details = ll_enable_tree_details

    def set_qu_n_estimators(self,qu_n_estimators):
        self.qu_n_estimators = qu_n_estimators

    def set_quantiles(self,quantiles):
        self.quantiles = quantiles

    def set_regression_splitting(self,regression_splitting):
        self.regression_splitting = regression_splitting

    def set_qu_equalize_cluster_weights(self,qu_equalize_cluster_weights):
        self.qu_equalize_cluster_weights = qu_equalize_cluster_weights

    def set_qu_sample_fraction(self,qu_sample_fraction):
        self.qu_sample_fraction = qu_sample_fraction

    def set_qu_mtry(self,qu_mtry):
        self.qu_mtry = qu_mtry

    def set_samples(self,samples):
        self.samples = samples
        quantiles = [i / samples for i in range(1 + samples)]

        quantiles[0] = 0 + 1 / (samples * 10)
        quantiles[-1] = 1 - 1 / (samples * 10)
        self.quantiles = quantiles

    def set_qu_min_node_size(self,qu_min_node_size):
        self.qu_min_node_size = qu_min_node_size

    def set_qu_honesty(self,qu_honesty):
        self.qu_honesty = qu_honesty

    def set_qu_honesty_fraction(self,qu_honesty_fraction):
        self.qu_honesty_fraction = qu_honesty_fraction

    def set_qu_honesty_prune_leaves(self,qu_honesty_prune_leaves):
        self.qu_honesty_prune_leaves = qu_honesty_prune_leaves

    def set_qu_alpha(self,qu_alpha):
        self.qu_alpha = qu_alpha

    def set_qu_imbalance_penalty(self,qu_imbalance_penalty):
        self.qu_imbalance_penalty = qu_imbalance_penalty

    def set_qu_n_jobs(self,qu_n_jobs):
        self.qu_n_jobs = qu_n_jobs

    def set_qu_seed(self,qu_seed):
        self.qu_seed = qu_seed

    def set_qu_enable_tree_details(self,qu_enable_tree_details):
        self.qu_enable_tree_details = qu_enable_tree_details

    def update_ll(self):
        self.ll_forest = GRFForestLocalLinearRegressor(n_estimators=self.ll_n_estimators,
                                                       ll_split_weight_penalty=self.ll_split_weight_penalty,
                                                       ll_split_lambda=self.ll_split_lambda,
                                                       ll_split_variables=self.ll_split_variables,
                                                       ll_split_cutoff=self.ll_split_cutoff,
                                                       equalize_cluster_weights=self.ll_equalize_cluster_weights,
                                                       sample_fraction=self.ll_sample_fraction, mtry=self.ll_mtry,
                                                       min_node_size=self.ll_min_node_size, honesty=self.ll_honesty,
                                                       honesty_fraction=self.ll_honesty_fraction,
                                                       honesty_prune_leaves=self.ll_honesty_prune_leaves,
                                                       alpha=self.ll_alpha, imbalance_penalty=self.ll_imbalance_penalty,
                                                       ci_group_size=self.ll_ci_group_size,
                                                       n_jobs=self.ll_n_jobs, seed=self.ll_seed,
                                                       enable_tree_details=self.ll_enable_tree_details)


    def update_qu(self):
        self.qu_forest = GRFForestQuantileRegressor(n_estimators=self.qu_n_estimators, quantiles=self.quantiles,
                                                    regression_splitting=self.regression_splitting,
                                                    equalize_cluster_weights=self.qu_equalize_cluster_weights,
                                                    sample_fraction=self.qu_sample_fraction, mtry=self.qu_mtry,
                                                    min_node_size=self.qu_min_node_size,
                                                    honesty=self.qu_honesty, honesty_fraction=self.qu_honesty_fraction,
                                                    honesty_prune_leaves=self.qu_honesty_prune_leaves,
                                                    alpha=self.qu_alpha,
                                                    imbalance_penalty=self.qu_imbalance_penalty,
                                                    n_jobs=self.qu_n_jobs, seed=self.qu_seed,
                                                    enable_tree_details=self.qu_enable_tree_details)

    def train_ll(self,X_train,y_train,update=True):
        if update:
            self.update_ll()

        self.ll_forest.fit(X_train,y_train)

    def train_qu(self, X_train, y_train, update=True):
        if update:
            self.update_qu()

        self.qu_forest.fit(X_train, y_train)

    def train(self,X_train,y_train,train_ll=True,train_qu=True,update_ll=True,update_qu=True):

        if train_ll:
            self.train_ll(X_train,y_train,update_ll)
        if train_qu:
            self.train_qu(X_train,y_train,update_qu)

    def predict_ll(self,x_test):
        result = self.ll_forest.predict(x_test)
        return result

    def predict_qu(self,x_test):
        raw_result = self.qu_forest.predict(x_test)

        out_results = {"x":[],"y_std":[],"y_mu":[]}

        for i in self.quantiles:
            out_results["cdf_%.2f" % i] = []

        for k in range(x_test.shape[0]):
            out_results["x"].append(x_test[k][0])
            out_results["y_std"].append(np.std(raw_result[k]))
            out_results["y_mu"].append(np.mean(raw_result[k]))
            for i in range(len(self.quantiles)):
                out_results[("cdf_%.2f" % self.quantiles[i])].append(raw_result[k][i])

        out_results = pd.DataFrame(out_results)
        return raw_result,out_results