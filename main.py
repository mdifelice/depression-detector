#!.venv/bin/python
from datetime import datetime
from getopt import getopt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from scipy.stats import ttest_ind
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, train_test_split, cross_validate
from sklearn.preprocessing import *
from typing import Callable
import importlib
import json
import numpy as np
import os
import pandas as pd
import re
import sys
import time

def print_message( message, eol = True, timestamp = True ):
	now = datetime.now()

	print( ( now.strftime( "[%Y-%m-%d %H:%M:%S] " ) if timestamp else "" ) + message, end = "\n" if eol else "", flush = True )

def get_file_path( file_name ):
	return os.path.dirname( __file__ ) + "/data/" + file_name

def get_metadata_setting( setting, default_value ):
	return metadata.get( setting, default_value ) if metadata else default_value

def get_column_metadata( dataset_id, column, field ):
	return get_metadata_setting( "datasets", {} ) \
	.get( dataset_id, {} ) \
	.get( column, {} ) \
	.get( field )

def get_target_column( dataset_id ):
	target_column = None

	for col in dataset.columns:
		if get_column_metadata( dataset_id, col, "target" ):
			target_column = col
			break

	return target_column

def get_formatted_metrics( metrics ):
	message = ""

	for key, value in metrics.items():
		message += ( ", " if message else "" ) + f"{key} {value:.4}"

	return "\033[" + ( "32" if metrics_are_successful( metrics ) else "31" ) + "m" + message + "\033[0m"

def get_current_time():
	return time.time()

def get_p_values( dataset_1, dataset_2 ):
	t_stats = {}
	p_values = {}

	for column in dataset_1:
		t_stats[ column ] = {}
		p_values[ column ] = None

		if pd.api.types.is_numeric_dtype( dataset_1[ column ] ) and pd.api.types.is_numeric_dtype( dataset_2[ column ] ):
			_, p_value = ttest_ind( dataset_1[ column ], dataset_2[ column ] )

			p_values[ column ] = p_value

	return p_values

def t_test( clusters ):
	count_p_values = 0
	good_p_values = 0
	columns_to_check = clusters[0].columns

	for i in range( len( clusters ) ):
		for j in range( len( clusters ) ):
			if j > i:
				cluster_1 = clusters[ i ]
				cluster_2 = clusters[ j ]

				p_values = get_p_values( cluster_1, cluster_2 )

				for column in columns_to_check:
					if column in p_values:
						count_p_values += 1
					
					try:
						if p_values[ column ] < 0.05:
							good_p_values += 1
					except:
						continue

	return good_p_values / count_p_values if count_p_values != 0 else .0

def get_model_metrics( model, X, y, time = None, fitted_model = None ):
	metrics = pd.Series()
	cb_time = 0

	if unsupervised:
		before_callback = lambda : model.predict( X ) if getattr( model, 'predict', False ) else fitted_model.labels_
		metrics_to_calculate = {
			"Silhouette" : lambda y : silhouette_score( X, y ),
			"DBI" : lambda y : davies_bouldin_score( X, y ),
			"Calinski-Harabasz" : lambda y : calinski_harabasz_score( X, y ),
			"T-Test" : t_test
		}
		after_callback = None
	else:
		min_cv_folds = y.value_counts().min()

		before_callback = lambda : cross_validate( model, X, y, cv = cross_validation_folds if cross_validation_folds <= min_cv_folds else min_cv_folds, scoring = [ "f1", "roc_auc", "accuracy", "precision", "recall" ], error_score = 'raise' )

		after_callback = lambda scores : sum( scores["fit_time"] )
		metrics_to_calculate = {
			"F1" : lambda scores : scores["test_f1"].mean(),
			"AUC" : lambda scores : scores["test_roc_auc"].mean(),
			"Accuracy" : lambda scores : scores["test_accuracy"].mean(),
			"Prec." : lambda scores : scores["test_precision"].mean(),
			"Recall" : lambda scores : scores["test_recall"].mean(),
		}

	for metric in metrics_to_calculate:
		metrics.loc[ metric ] = .0

	try:
		before_response = before_callback()

		for metric, callback in metrics_to_calculate.items():
			value = None

			try:
				value = callback( before_response )
			except Exception as e:
				if debug:
					print_message( f"Error retrieving metric {metric} for {model.__class__.__name__}: {e}" )

			metrics.loc[ metric ] = value

			cb_time = after_callback( before_response ) if after_callback else .0
	except Exception as e:
		if debug:
			print_message( f"Error retrieving metrics for {model.__class__.__name__}: {e}" )

	metrics.loc["TT (Sec)"] = time + cb_time if time else cb_time

	return metrics

def expand_model_names( model_names ):
	expanded_model_names = []

	for model_name in model_names:
		for available_model in available_models:
			if model_name in available_model:
				expanded_model_names.append( available_model )

	return expanded_model_names

def metrics_are_successful( metrics ):
	metric = None

	match tune_scoring:
		case "f1":
			metric = "F1"
		case "silhouette_score":
			metric = "Silhouette"

	return metrics[ metric ] >= metric_acceptance_threshold if metric else False

def experiment( title, dataset, tune = False, **pycaret_setup_args ):
	success = False
	metrics = pd.DataFrame()

	print_message( title + "...", eol = debug )

	if not debug:
		original_stdout = sys.stdout
		original_stderr = sys.stderr

		sys.stdout = open( os.devnull, "w" )
		sys.stderr = open( os.devnull, "w" )

	if pycaret_setup_args:
		if unsupervised:
			pycaret_experiment = ClusteringExperiment()

			pycaret_experiment.setup(
				data = dataset,
				session_id = random_seed
			)

			print_message( "No unsupervised support for PyCaret" )
		elif target_column:
			pycaret_experiment = ClassificationExperiment()

			pycaret_experiment.setup(
				data = dataset,
				target = target_column,
				session_id = random_seed,
				train_size = 1 - test_ratio,
				verbose = debug > 1,
				**pycaret_setup_args
			)

			pycaret_available_models = pycaret_experiment.models()

			if tune:
				for index in range( len( pycaret_available_models ) ):
					row = pycaret_available_models.iloc[ index ]

					if row["Reference"] in selected_models:
						start_time = get_current_time()

						model = pycaret_experiment.tune_model(
							pycaret_experiment.create_model(
								pycaret_available_models.index[ index ],
								verbose = debug > 1
							),
							optimize = tune_scoring,
							n_iter = tune_iterations,
							verbose = debug > 1
						)

						model_metrics = pycaret_experiment.pull().loc["Mean"]
						model_metrics["TT (Sec)"] = get_current_time() - start_time
						model_metrics.name = model

						metrics = pd.concat( [ metrics, model_metrics.to_frame().T ] )
			else:
				selected_models_ids = []

				for index in range( len( pycaret_available_models ) ):
					row = pycaret_available_models.iloc[ index ]

					if row["Reference"] in selected_models:
						selected_models_ids.append( pycaret_available_models.index[ index ] )

				pycaret_experiment.compare_models(
					verbose = debug > 1,
					include = selected_models_ids
				)

				models_metrics = pycaret_experiment.pull()

				for index in range( len( models_metrics ) ):
					model = pycaret_experiment.create_model( 
						models_metrics.index[ index ],
						verbose = debug > 1
					)

					model_metrics = models_metrics.iloc[ index ]
					model_metrics.name = model

					metrics = pd.concat( [ metrics, model_metrics.to_frame().T ] )
		else:
			print_message( "No target column, skipping supervised analysis." )
	else:
		if unsupervised:
			X_train = dataset
			y_train = None

			X_test = dataset
			y_test = None
		elif target_column:
			X = dataset.drop( target_column, axis = 1 )
			y = dataset[ target_column ]

			X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_ratio, random_state = random_seed, stratify = y )
		else:
			X_train = None

		if X_train is not None:
			for model_name, settings in available_models.items():
				if model_name in selected_models:
					module_name, class_name = model_name.rsplit( ".", 1 )

					module = importlib.import_module( module_name )

					constructor_settings = settings.get( "constructor", {} )

					args = constructor_settings.get( "arguments", {} )

					if constructor_settings.get( "random_state" ):
						args["random_state"] = random_seed

					start_time = get_current_time()

					model = getattr( module, class_name )( **args )

					if tune:
						param_grids = settings.get( "param_grids" )

						if not param_grids:
							param_grid = settings.get( "param_grid" )

							if param_grid:
								param_grids = []

								param_grids.append( param_grid )

						if param_grids:
							for param_grid in param_grids:
								cv = KFold( random_state = random_seed, shuffle = True, n_splits = cross_validation_tune_folds )

								match tune_scoring:
									case 'silhouette_score':
										cv_scoring = make_scorer( silhouette_score )
									case _:
										cv_scoring = tune_scoring

								if tune_iterations:
									search = RandomizedSearchCV( model, param_grid, n_iter = tune_iterations, cv = cv, scoring = cv_scoring, random_state = random_seed, verbose = 0 if debug < 2 else 4, n_jobs = None if not turbo else -1 )
								else:
									search = GridSearchCV( model, param_grid, cv = cv, scoring = cv_scoring, verbose = 0 if debug < 2 else 4, n_jobs = None if not turbo else -1 )

								if ( debug ):
									print_message( f"Fitting {model.__class__.__name__}..." )

								fitted_model = search.fit( X_train, y_train )

								model = search.best_estimator_
						else:
							model = None
					else:
						if ( debug ):
							print_message( f"Fitting {model.__class__.__name__}..." )

						fitted_model = model.fit( X_train, y_train )

					if model:
						model_metrics = get_model_metrics( model, X_test, y_test, get_current_time() - start_time, fitted_model )
						model_metrics.name = model

						metrics = pd.concat( [ metrics, model_metrics.to_frame().T ] )
		else:
			print_message( "No target column, skipping supervised analysis." )

	if not debug:
		sys.stdout = original_stdout
		sys.stderr = original_stderr

	best_model = None

	if metrics.empty:
		message = "No metrics found."
	else:
		best_model_index = 0
		models_to_print = {}
		max_value = 0

		if unsupervised:
			sort_by = "Calinski-Harabasz"
		else:
			sort_by = "AUC"

		metrics.sort_values( by = sort_by, ascending = False, inplace = True )

		for index in range( len( metrics ) ):
			if print_all:
				models_to_print[ metrics.index[ index ] ] = metrics.iloc[ index ]

			model_metrics = metrics.iloc[ index ]

			if metrics_are_successful( model_metrics ):
				success = True

				if model_metrics[ sort_by ] > max_value:
					max_value = model_metrics[ sort_by ]
					best_model_index = index

		message = ""
		best_model = metrics.index[ best_model_index ]

		if not print_all:
			models_to_print[ metrics.index[ best_model_index ] ] = metrics.iloc[ best_model_index ]

		for model, model_metrics in models_to_print.items():
			message += ( "" if len( models_to_print ) == 1 else "\n" ) + f"{model if debug else model.__class__.__name__}: {get_formatted_metrics( model_metrics )}"

	print_message( ( "" if debug else " " ) + message, timestamp = debug )

	return success

metadata = None
allowed_extensions = [ "csv", "xlsx" ]
random_seed = 123
test_ratio = .3
row_acceptance_threshold = .75
column_acceptance_threshold = .25
max_ohe_unique_values = 10
correlation_acceptance_threshold = .6
oversampling_threshold = 0
tune_iterations = 10
tune = False
cross_validation_folds = 5
cross_validation_tune_folds = 5
force_tuning = False
train = False
pycaret = None
debug = False
unsupervised = False
print_all = False
turbo = False
selected_datasets = list( map( str, range( 1, 11 ) ) )
selected_models = None
excluded_models = None
print_preprocessing = False
scaler_type = "standard"

available_options = {
	"a" : {
		"variable" : "print_all",
	},
	"c" : {
		"variable" : "print_preprocessing",
	},
	"d" : {
		"has_value" : lambda x: list( map( str.strip, x.split( "," ) ) ),
		"variable" : "selected_datasets",
	},
	"e" : {
		"variable" : "debug",
		"has_value" : int,
	},
	"f" : {
		"variable" : "force_tuning",
	},
	"i" : {
		"variable" : "cross_validation_tune_folds",
		"has_value" : int,
	},
	"l" : {
		"variable" : "scaler_type",
		"has_value" : True
	},
	"m" : {
		"has_value" : lambda x: list( map( str.strip, x.split( "," ) ) ),
		"variable" : "selected_models",
	},
	"n" : {
		"variable" : "tune",
	},
	"o" : {
		"has_value" : int,
		"variable" : "oversampling_threshold",
	},
	"p" : {
		"has_value" : True,
		"variable" : "pycaret",
	},
	"r" : {
		"variable" : "turbo",
	},
	"s" : {
		"variable" : "test_ratio",
		"has_value" : float,
	},
	"t" : {
		"variable" : "train",
	},
	"u" : {
		"variable" : "unsupervised",
	},
	"v" : {
		"variable" : "cross_validation_folds",
		"has_value" : int,
	},
	"w" : {
		"variable" : "random_seed",
		"has_value" : int,
	},
	"x" : {
		"has_value" : lambda x: list( map( str.strip, x.split( "," ) ) ),
		"variable" : "excluded_models",
	},
	"y" : {
		"has_value" : int,
		"variable" : "tune_iterations",
	},
}

try:
	options, extra_args = getopt( sys.argv[ 1: ], "".join( map( lambda x: x + ( ":" if available_options.get( x, {} ) .get( "has_value" ) else "" ), available_options.keys() ) ) )

	for key, value in options:
		option_settings = available_options.get( key[ 1: ], {} )

		if option_settings:
			variable = option_settings.get( "variable" )

			if variable:
				parser = option_settings.get( "has_value", lambda x: True )

				if isinstance( parser, Callable ):
					value = parser( value )

				globals()[ variable ] = value
except Exception as e:
	print( f"Error parsing arguments: {e}" )
	sys.exit(1)

if debug:
	print_preprocessing = True
	print_all = True

with open( get_file_path( "metadata.json" ) ) as metadata_file:
	metadata = json.load( metadata_file )
	metadata_file.close()

available_models = get_metadata_setting( "unsupervised_algorithms" if unsupervised else "supervised_algorithms", {} )

if selected_models is None:
	selected_models = list( available_models.keys() )
else:
	selected_models = expand_model_names( selected_models )

if excluded_models:
	excluded_models = expand_model_names( excluded_models )

	for model in excluded_models:
		if model in selected_models:
			selected_models.remove( model )

if unsupervised:
	tune_scoring = "silhouette_score"
	metric_acceptance_threshold = .7
else:
	tune_scoring = "f1"
	metric_acceptance_threshold = .7

for dataset_id in selected_datasets:
	dataset = None
	file_path = None
	extension = None

	for allowed_extension in allowed_extensions:
		maybe_file_path = get_file_path( f"dataset-{dataset_id}.{allowed_extension}" )

		if os.path.exists( maybe_file_path ):
			file_path = maybe_file_path
			extension = allowed_extension

	if file_path: 
		if extension == "csv":
			if "6" == dataset_id:
				separator = ";"
			else:
				separator = ","

			dataset = pd.read_csv( file_path, sep = separator )
		elif extension == "xlsx":
			dataset = pd.read_excel( file_path )

	if dataset is not None:
		print_message( f"Dataset {dataset_id} {dataset.shape}..." )

		target_column = get_target_column( dataset_id )

		if print_preprocessing:
			print_message( "Preprocessing..." )

		preprocessed_dataset = dataset.copy()

		rows_to_remove = []

		preprocessed_dataset = preprocessed_dataset.rename( columns = lambda x: x.strip() )

		for col in preprocessed_dataset.columns:
			if get_column_metadata( dataset_id, col, "allow_na" ):
				preprocessed_dataset[ col ] = preprocessed_dataset[ col ].fillna( "" )
			else:
				preprocessed_dataset[ col ] = preprocessed_dataset[ col ].apply( lambda x : x.strip() if isinstance( x, str ) else x ).replace( "", np.nan )

		for index in preprocessed_dataset.index:
			if ( preprocessed_dataset.iloc[ index ].isna().sum() / len( preprocessed_dataset.iloc[ index ] ) ) > row_acceptance_threshold:
				rows_to_remove.append( index )

		preprocessed_dataset = preprocessed_dataset.drop( index = rows_to_remove )

		columns_to_remove = []

		for col in preprocessed_dataset.columns:
			column_to_remove = None

			if col != target_column and preprocessed_dataset[ col ].isnull().sum() / len( preprocessed_dataset[ col ] ) > column_acceptance_threshold:
				column_to_remove = col
			elif col == target_column and unsupervised:
				column_to_remove = col

				target_column = None

			if column_to_remove:
				columns_to_remove.append( column_to_remove )

		preprocessed_dataset = (
			preprocessed_dataset
				.drop( columns = columns_to_remove )
				.dropna()
		)

		if print_preprocessing:
			print_message( f"Dataset #{dataset_id} dimension reduction from {dataset.shape} to {preprocessed_dataset.shape}.")

			print_message( "Engineering..." )

		engineered_dataset = preprocessed_dataset.copy()

		# Remove columns
		for col in engineered_dataset.columns:
			if get_column_metadata( dataset_id, col, "removable" ):
				engineered_dataset = engineered_dataset.drop( columns = [ col ] )

		for col in engineered_dataset.columns:
			target_definition = get_column_metadata( dataset_id, col, "target" )

			if target_definition and type( target_definition ) is list:
				unique_values = engineered_dataset[ col ].unique()

				unique_values = sorted( unique_values, key = ( lambda x: str( x ) ) )

				if len( unique_values ) != 2 or unique_values[0] != 0 or unique_values[1] != 1:
					engineered_dataset[ col ] = engineered_dataset[ col ].apply( lambda x: 1 if x in target_definition else 0 )

		unprocessed_dataset = engineered_dataset.copy()

		for col in engineered_dataset.select_dtypes( include=[ "object" ] ):
			if not get_column_metadata( dataset_id, col, "multiple" ):
				order = get_column_metadata( dataset_id, col, "order" )

				if order is not None:
					if not isinstance( order, bool ):
						mapping = { value: i for i, value in enumerate( order ) }
					else:
						mapping = None
				else:
					unique_values = engineered_dataset[ col ].unique()

					mapping = { value: i for i, value in enumerate( unique_values ) }

				if mapping is not None:
					engineered_dataset[ col ] = engineered_dataset[ col ].map( mapping )
			else:
				# One-hot encode multiple columns
				values = set()

				for row in engineered_dataset[ col ]:
					if isinstance( row, str ):
						values.update( filter( len, [ x.strip() for x in row.split( "," ) ] ) )

				for value in values:
					engineered_dataset[ col + "_" + value ] = engineered_dataset[ col ].apply( lambda x: 1 if isinstance( x, str ) and value in x.split( "," ) else 0 )

				engineered_dataset = engineered_dataset.drop( columns = [ col ] )

		# Transform date columns to numeric values (e.g., timestamps)
		for col in engineered_dataset.select_dtypes( include = [ "datetime64" ] ):
			engineered_dataset[ col ] = pd.to_numeric(engineered_dataset[ col ] )

		# Apply scaling
		match scaler_type:
			case 'standard':
				scaler = StandardScaler()
			case 'minmax':
				scaler = MinMaxScaler()
			case 'robust':
				scaler = RobustScaler()
			case _:
				scaler = None

		if scaler:
			if target_column:
				X = engineered_dataset.drop( target_column, axis = 1 )
				y = engineered_dataset[ target_column ]

				scaler.set_output( transform = "pandas" )
				X_scaled = scaler.fit_transform( X )

				engineered_dataset = pd.concat( [ X_scaled, pd.Series( y, name = target_column ) ], axis = 1 )
			else:
				engineered_dataset[ engineered_dataset.columns ] = scaler.fit_transform( engineered_dataset[ engineered_dataset.columns ] )

		# Correlation analysis
		correlation_matrix = engineered_dataset.corr().abs()

		upper = correlation_matrix.where( np.triu( np.ones( correlation_matrix.shape ), k = 1).astype( bool ) )

		to_drop = set()

		for col1 in upper.columns:
			if col1 != target_column:
				for col2 in upper.index:
					if col2 != target_column:
						if upper.loc[ col2, col1 ] > correlation_acceptance_threshold:
							if target_column:
								# Determine which column to drop: the one with the lower relationship with the target column
								relationship_with_target = {
									col1: upper.loc[ col1, target_column ],
									col2: upper.loc[ col2, target_column ]
								}

								col_to_drop = min( relationship_with_target, key = relationship_with_target.get )
							else:
								col_to_drop = col1

							to_drop.add( col_to_drop )

		engineered_dataset = engineered_dataset.drop( columns = list( to_drop ) )

		if print_preprocessing:
			print_message( f"Correlation analysis on dataset #{dataset_id}, removing {len(to_drop)} column/s: {sorted(to_drop)}" )

		# Identify categorical columns for one-hot encoding (excluding target)
		categorical_cols_to_encode = []

		for col in engineered_dataset.columns:
			if col != target_column and get_column_metadata( dataset_id, col, "order" ) is None:
				unique_values = engineered_dataset[ col ].nunique()

				if unique_values < max_ohe_unique_values and unique_values > 2: # Check for less than 10 unique values
					categorical_cols_to_encode.append( ( col, unique_values ) )

		categorical_cols_to_encode.sort( key = lambda item: item[0] )

		if categorical_cols_to_encode:
			message = ""
			columns = []

			for col in categorical_cols_to_encode:
				columns.append( col[0] )
				message += ( ", " if message else "" ) + f"{col[0]} ({col[1]} valores únicos)"

			if print_preprocessing:
				print_message( f"One-hot encoding on dataset #{dataset_id}, {len(categorical_cols_to_encode)} column/s: {message}" )

			engineered_dataset = pd.get_dummies( engineered_dataset, columns = columns, dummy_na = False, dtype = int ) # dummy_na=False to not create a column for NaN

		# Drop duplicate rows
		engineered_dataset = engineered_dataset.drop_duplicates()

		if print_preprocessing:
			print_message( f"Dataset #{dataset_id} dimension modification after engineering from {preprocessed_dataset.shape} to {engineered_dataset.shape}.")

		column_renamer = lambda x: re.sub( "[^A-Za-z0-9_]+", "_", x )
		engineered_dataset = engineered_dataset.rename( columns = column_renamer )
		unprocessed_dataset = unprocessed_dataset.rename( columns = column_renamer )

		if target_column is not None:
			target_column = column_renamer( target_column )

			# todo check Nan values
			X = engineered_dataset.drop( target_column, axis = 1 )
			y = engineered_dataset[ target_column ]

			oversampling = len(y) < oversampling_threshold

			if oversampling:
				rs = SMOTE( random_state = random_seed )
			else:
				rs = RandomUnderSampler( random_state = random_seed )

			X, y = rs.fit_resample( X, y )

			processed_dataset = pd.concat( [ X, pd.Series( y, name = target_column ) ], axis = 1 )

			if print_preprocessing:
				print_message( f"Dataset #{dataset_id} dimension modification after balancing from {engineered_dataset.shape} to {processed_dataset.shape}.")
		else:
			processed_dataset = engineered_dataset

		if train:
			match pycaret:
				case "only_training":
					experiments_settings = {
						"Pycaret only training" : {
							"dataset" : processed_dataset,
							"pycaret_setup_args" : {
								"preprocess" : False
							}
						},
						"Pycaret only training and tuning" : {
							"dataset" : processed_dataset,
							"tune" : True,
							"pycaret_setup_args" : {
								"preprocess" : False
							}
						}
					}
				case "complete":
					experiments_settings = {
						"Pycaret" : {
							"dataset" : unprocessed_dataset,
							"pycaret_setup_args" : {
								"preprocess" : True
							}
						},
						"Pycaret with normalization" : {
							"dataset" : unprocessed_dataset,
							"pycaret_setup_args" : {
								"normalize" : True
							}
						},
						"Pycaret with balancing" : {
							"dataset" : unprocessed_dataset,
							"pycaret_setup_args" : {
								"fix_imbalance" : True
							}
						},
						"Pycaret with normalization and balancing" : {
							"dataset" : unprocessed_dataset,
							"pycaret_setup_args" : {
								"normalize" : True,
								"fix_imbalance" : True
							}
						},
						"Pycaret with normalization, balancing and tuning" : {
							"dataset" : unprocessed_dataset,
							"tune" : True,
							"pycaret_setup_args" : {
								"normalize" : True,
								"fix_imbalance" : True
							}
						}
					}
				case _:
					experiments_settings = {
						"Custom" : {
							"dataset" : processed_dataset,
						},
						"Custom with tuning" : {
							"dataset" : processed_dataset,
							"tune" : True,
						},
					}

			for key, value in experiments_settings.items():
				experiment_tune = value.get( "tune", False )

				if not experiment_tune or tune:
					if experiment( key, value.get( "dataset" ), tune = experiment_tune, **value.get( "pycaret_setup_args", {} ) ):
						if not force_tuning:
							break

print_message( "Finished." )
