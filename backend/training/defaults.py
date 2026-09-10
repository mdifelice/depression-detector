from copy import deepcopy

from models import ModelConfig

LDA = "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
QDA = "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
ADA = "sklearn.ensemble._weight_boosting.AdaBoostClassifier"
ExTREES = "sklearn.ensemble._forest.ExtraTreesClassifier"
GB = "sklearn.ensemble._gb.GradientBoostingClassifier"
RF = "sklearn.ensemble._forest.RandomForestClassifier"
GP = "sklearn.gaussian_process._gpc.GaussianProcessClassifier"
LR = "sklearn.linear_model._logistic.LogisticRegression"
RIDGE = "sklearn.linear_model._ridge.RidgeClassifier"
NB = "sklearn.naive_bayes.GaussianNB"
KNN = "sklearn.neighbors._classification.KNeighborsClassifier"
SGD = "sklearn.linear_model._stochastic_gradient.SGDClassifier"
DT = "sklearn.tree._classes.DecisionTreeClassifier"
XGB = "xgboost.sklearn.XGBClassifier"
MLP = "sklearn.neural_network._multilayer_perceptron.MLPClassifier"
HGB = "sklearn.ensemble.HistGradientBoostingClassifier"
LINEAR_SVC = "sklearn.svm.LinearSVC"

DEFAULT_PARAM_GRIDS: dict[str, dict | list[dict]] = {
    LDA: [
        {"solver": ["svd"]},
        {
            "solver": ["lsqr", "eigen"],
            "shrinkage": [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        },
    ],
    QDA: {
        "reg_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    ADA: {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.5, 1],
    },
    ExTREES: {
        "n_estimators": [10, 100, 200, 500, 1000],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8, None],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
    GB: {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "loss": ["log_loss", "deviance", "exponential"],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "criterion": ["friedman_mse", "squared_error"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [3, 5, 8],
        "max_features": ["sqrt", "log2", None],
    },
    RF: {
        "n_estimators": [100, 200, 500, 1000],
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
        "bootstrap": [True],
    },
    GP: {
        "optimizer": ["fmin_l_bfgs_b"],
        "n_restarts_optimizer": [0, 1, 2, 5],
        "max_iter_predict": [100, 200, 500],
    },
    LR: {
        "penalty": ["l1", "l2", "elasticnet", None],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear", "saga"],
        "class_weight": [None, "balanced"],
    },
    RIDGE: {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        "class_weight": [None, "balanced"],
    },
    NB: {
        "var_smoothing": [
            1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01,
            0.1, 0.2, 0.5, 1.0,
        ],
    },
    KNN: {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2],
    },
    SGD: {
        "loss": ["hinge", "log_loss", "modified_huber"],
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": [0.000000001, 0.0001, 0.001, 0.01, 0.1, 0.99],
        "l1_ratio": [0.000000001, 0.0001, 0.01, 0.15, 0.5, 0.85],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "fit_intercept": [True, False],
        "power_t": [0.5, 0.1],
        "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "max_iter": [1000, 2000],
    },
    DT: {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    XGB: {
        "n_estimators": [100, 200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.5, 1.0],
        "reg_alpha": [0, 0.001, 0.01, 0.1],
        "reg_lambda": [0, 0.001, 0.01, 0.1],
    },
    MLP: {
        "hidden_layer_sizes": [[50], [100], [50, 50], [100, 50, 25]],
        "activation": ["relu", "tanh", "logistic"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "max_iter": [200, 500],
    },
    HGB: {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [15, 31],
        "max_depth": [None, 10],
        "min_samples_leaf": [1, 5],
        "l2_regularization": [0.0, 0.1],
    },
    LINEAR_SVC: {
        "C": [0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "tol": [1e-4, 1e-3],
    },
}

DEFAULT_CONSTRUCTOR_PARAMS: dict[str, dict] = {
    ADA: {"estimator": None},
}

SEED_ALGORITHMS: set[str] = {
    LINEAR_SVC,
    HGB,
    MLP,
    XGB,
    DT,
    SGD,
    RIDGE,
    LR,
    GP,
    RF,
    ExTREES,
    ADA,
}


def build_default_models() -> dict[str, ModelConfig]:
    return {
        model_path: ModelConfig(
            constructor_params=deepcopy(DEFAULT_CONSTRUCTOR_PARAMS.get(model_path, {})),
            param_grid=deepcopy(DEFAULT_PARAM_GRIDS.get(model_path, {})),
        )
        for model_path in (
            LDA, QDA, ADA, ExTREES, GB, RF, GP, LR, RIDGE, NB, KNN,
            SGD, DT, XGB, MLP, HGB, LINEAR_SVC,
        )
    }