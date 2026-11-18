from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    email: str
    name: str
    picture: str


class CategoricalColumnConfig(BaseModel):
    order: list[str]
    ordinal: bool = True


class DatasetMetadata(BaseModel):
    id: str
    filename: str
    ignore_columns: list[str] = []
    nullable_columns: list[str] = []
    target_column: Optional[str] = None
    positive_values: list[str] = []
    categorical_columns: dict[str, CategoricalColumnConfig] = {}
    multi_value_columns: list[str] = []
    uploaded_at: str
    uploaded_by: str


class DatasetMetadataUpdate(BaseModel):
    ignore_columns: Optional[list[str]] = None
    nullable_columns: Optional[list[str]] = None
    target_column: Optional[str] = None
    positive_values: Optional[list[str]] = None
    categorical_columns: Optional[dict[str, CategoricalColumnConfig]] = None
    multi_value_columns: Optional[list[str]] = None


class DatasetInfo(BaseModel):
    id: str
    filename: str
    uploaded_at: str
    uploaded_by: str
    configured: bool = False
    trained: bool = False


class DatasetColumnsResponse(BaseModel):
    columns: list[str]
    sample: list[list[str]]
    unique_value_counts: list[int]
    null_counts: list[int]
    row_count: int


AVAILABLE_MODELS = [
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
    "sklearn.ensemble._weight_boosting.AdaBoostClassifier",
    "sklearn.ensemble._forest.ExtraTreesClassifier",
    "sklearn.ensemble._gb.GradientBoostingClassifier",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.gaussian_process._gpc.GaussianProcessClassifier",
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.linear_model._ridge.RidgeClassifier",
    "sklearn.naive_bayes.GaussianNB",
    "sklearn.neighbors._classification.KNeighborsClassifier",
    "sklearn.linear_model._stochastic_gradient.SGDClassifier",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "xgboost.sklearn.XGBClassifier",
    "sklearn.neural_network._multilayer_perceptron.MLPClassifier",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "sklearn.svm.LinearSVC",
]

SCALING_TYPES = ["standard", "minmax", "robust"]


class ModelConfig(BaseModel):
    constructor_params: dict = {}
    param_grid: dict | list[dict] = {}


class TrainingSettings(BaseModel):
    tune: bool = False
    cross_validation_folds: int = 5
    cross_validation_tune_folds: int = 5
    scaling_type: str = "standard"
    use_sampling: bool = False
    oversampling_threshold: Optional[int] = None
    test_ratio: float = 0.2
    turbo: bool = False
    random_seed: int = 123
    tune_iterations: int = 50
    timeout: int = 0
    row_acceptance_threshold: float = 0.75
    column_acceptance_threshold: float = 0.25
    max_ohe_unique_values: int = 10
    max_sortable_values: int = 25
    correlation_acceptance_threshold: float = 0.8
    use_shap: bool = True
    use_lime: bool = True
    use_anchor: bool = True
    selected_models: list[str] = AVAILABLE_MODELS.copy()
    models: dict[str, ModelConfig] = {
        m: ModelConfig() for m in AVAILABLE_MODELS
    }


class TrainingSettingsUpdate(BaseModel):
    tune: Optional[bool] = None
    cross_validation_folds: Optional[int] = None
    cross_validation_tune_folds: Optional[int] = None
    scaling_type: Optional[str] = None
    use_sampling: Optional[bool] = None
    oversampling_threshold: Optional[int] = None
    test_ratio: Optional[float] = None
    turbo: Optional[bool] = None
    random_seed: Optional[int] = None
    tune_iterations: Optional[int] = None
    timeout: Optional[int] = None
    row_acceptance_threshold: Optional[float] = None
    column_acceptance_threshold: Optional[float] = None
    max_ohe_unique_values: Optional[int] = None
    max_sortable_values: Optional[int] = None
    correlation_acceptance_threshold: Optional[float] = None
    use_shap: Optional[bool] = None
    use_lime: Optional[bool] = None
    use_anchor: Optional[bool] = None
    selected_models: Optional[list[str]] = None
    models: Optional[dict[str, ModelConfig]] = None


class JobStatus(BaseModel):
    dataset_id: str
    status: str
    progress: float
    current_step: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ModelResult(BaseModel):
    model_name: str
    f1: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    auc: Optional[float] = None
    ks: Optional[float] = None
    lift_at_10: Optional[float] = None
    train_time: Optional[float] = None
    is_best: bool = False
    roc_chart: Optional[str] = None
    confusion_matrix_chart: Optional[str] = None
    error: Optional[str] = None


class SavedModelInfo(BaseModel):
    dataset_id: str
    filename: str
    model_name: str
    metrics: dict
    has_explanation: bool = False


class PredictionRequest(BaseModel):
    values: dict[str, str]


class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    label: str
    explain_methods: list[str] = []


class ColumnValuesResponse(BaseModel):
    column: str
    unique_values: list[str]


class SampleRowsResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
