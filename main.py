#!.venv/bin/python
from datetime import datetime
from getopt import getopt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pycaret.classification import *
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
	return "\033[" + ( "32" if metrics_are_successful( metrics ) else "31" ) + "m" + f"F1 {metrics["F1"]:.4f}, AUC {metrics["AUC"]:.4f}, Accuracy {metrics["Accuracy"]:.4f}, Precision {metrics["Prec."]:.4f}, Recall {metrics["Recall"]:.4f}" + ( f", TT {metrics["TT (Sec)"]:.4f}" if not np.isnan( metrics["TT (Sec)"] ) else "" ) + "\033[0m"

def get_current_time():
	return time.time()

def get_model_metrics( model, X, y, time = None ):
	y_pred = model.predict( X )

	try:
		row_auc_score_metric = roc_auc_score( y, y_pred )
	except:
		row_auc_score_metric = 0

	return pd.Series( {
		"F1" : f1_score( y, y_pred ),
		"AUC" : row_auc_score_metric,
		"Accuracy" : accuracy_score( y, y_pred ),
		"Prec." : precision_score( y, y_pred ),
		"Recall" : recall_score( y, y_pred ),
		"TT (Sec)" : time,
	} )

def expand_model_names( model_names ):
	expanded_model_names = []

	for model_name in model_names:
		for available_model in available_models:
			if model_name in available_model:
				expanded_model_names.append( available_model )

	return expanded_model_names

def metrics_are_successful( metrics ):
	return metrics["F1"] >= f1_acceptance_threshold

def experiment( title, dataset, tune = False, validation_dataset = None, **pycaret_setup_args ):
	success = False

	if ( unsupervised ):
		pass
		#todo
	else:
		print_message( title + "...", eol = False )

		metrics = pd.DataFrame()

		if not verbose:
			original_stdout = sys.stdout
			original_stderr = sys.stdout
			sys.stdout = open( os.devnull, "w" )
			sys.stderr = open( os.devnull, "w" )

		if pycaret_setup_args:
			setup(
				data = dataset,
				target = target_column,
				session_id = random_seed,
				train_size = 1 - test_ratio,
				verbose = verbose,
				**pycaret_setup_args
			)

			pycaret_available_models = models()

			if tune:
				for index in range( len( pycaret_available_models ) ):
					row = pycaret_available_models.iloc[ index ]

					if row["Reference"] in selected_models:
						start_time = get_current_time()

						model = tune_model(
							create_model(
								pycaret_available_models.index[ index ],
								verbose = verbose
							),
							optimize = "f1",
							n_iter = tune_iterations,
							verbose = verbose
						)

						model_metrics = pull().loc["Mean"]
						model_metrics["TT (Sec)"] = get_current_time() - start_time
						model_metrics.name = model

						metrics = pd.concat( [ metrics, model_metrics.to_frame().T ] )
			else:
				selected_models_ids = []

				for index in range( len( pycaret_available_models ) ):
					row = pycaret_available_models.iloc[ index ]

					if row["Reference"] in selected_models:
						selected_models_ids.append( pycaret_available_models.index[ index ] )

				compare_models(
					verbose = verbose,
					include = selected_models_ids
				)

				models_metrics = pull()

				for index in range( len( models_metrics ) ):
					model = create_model( 
						models_metrics.index[ index ],
						verbose = verbose
					)

					model_metrics = models_metrics.iloc[ index ]
					model_metrics.name = model

					metrics = pd.concat( [ metrics, model_metrics.to_frame().T ] )
		else:
			X = dataset.drop( target_column, axis = 1 )
			y = dataset[ target_column ]

			X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_ratio, random_state = random_seed )

			for model_name, settings in available_models.items():
				if model_name in selected_models:
					module_name, class_name = model_name.rsplit( '.', 1 )

					module = importlib.import_module( module_name )

					constructor_settings = settings.get( "constructor", {} )

					args = constructor_settings.get( "arguments", {} )

					if constructor_settings.get( "random_state" ):
						args["random_state"] = random_seed

					model = getattr( module, class_name )( **args )

					start_time = get_current_time()

					if tune:
						param_grids = settings.get( "param_grids" )

						if not param_grids:
							param_grid = settings.get( "param_grid" )

							if param_grid:
								param_grids = []

								param_grids.append( param_grid )

						if param_grids:
							for param_grid in param_grids:
								cv = KFold( random_state = random_seed, shuffle = True )

								if tune_iterations:
									search = RandomizedSearchCV( model, param_grid, n_iter = tune_iterations, cv = cv, scoring = "roc_auc", random_state = random_seed, verbose = 0 if not verbose else 4, n_jobs = None if not turbo else -1 )
								else:
									search = GridSearchCV( model, param_grid, cv = cv, scoring = "roc_auc", verbose = 0 if not verbose else 4, n_jobs = None if not turbo else -1 )

								if ( verbose ):
									print_message( f"Fitting {model.__class__.__name__}..." )

								search.fit( X_train, y_train )

								model = search.best_estimator_
						else:
							model = None
					else:
						if ( verbose ):
							print_message( f"Fitting {model.__class__.__name__}..." )

						model.fit( X_train, y_train )

					if model:
						model_metrics = get_model_metrics( model, X_test, y_test, get_current_time() - start_time )
						model_metrics.name = model

						metrics = pd.concat( [ metrics, model_metrics.to_frame().T ] )

		if not verbose:
			sys.stdout = original_stdout
			sys.stderr = original_stderr

		best_model = None

		if metrics.empty:
			message = "No metrics found."
		else:
			best_model_index = 0
			max_auc = 0
			models_to_print = {}

			metrics.sort_values( by = "AUC", ascending = False, inplace = True )

			for index in range( len( metrics ) ):
				if print_all:
					models_to_print[ metrics.index[ index ] ] = metrics.iloc[ index ]

				model_metrics = metrics.iloc[ index ]

				if metrics_are_successful( model_metrics ):
					success = True

					if model_metrics["AUC"] > max_auc:
						max_auc = model_metrics["AUC"]
						best_model_index = index

			message = ""
			best_model = metrics.index[ best_model_index ]

			if not print_all:
				models_to_print[ metrics.index[ best_model_index ] ] = metrics.iloc[ best_model_index ]

			for model, model_metrics in models_to_print.items():
				message += ( " " if len( models_to_print ) == 1 else "\n" ) + f"{model.__class__.__name__}: {get_formatted_metrics( model_metrics )}"

		print_message( message, timestamp = False )

		if success and best_model is not None and validation_dataset is not None:
			X = validation_dataset.drop( target_column, axis = 1 )
			y = validation_dataset[ target_column ]

			validation_metrics = get_model_metrics( best_model, X, y )

			if validation_metrics.empty:
				message = "No validation metrics"
			else:
				message = f"Validation: {get_formatted_metrics( validation_metrics )}"

			print_message( message )
	
	return success

metadata = None
allowed_extensions = [ "csv", "xlsx" ]
random_seed = 123
validation_ratio = .2
test_ratio = .3
row_acceptance_threshold = .75
column_acceptance_threshold = .25
max_ohe_unique_values = 10
f1_acceptance_threshold = .7
correlation_acceptance_threshold = .6
oversampling_threshold = 0
tune_iterations = 100

try:
	options, extra_args = getopt( sys.argv[ 1: ], "d:tp:vo:uaerm:x:" )
except Exception as e:
	print( f"Error parsing arguments: {e}" )
	sys.exit(1)

train = False
pycaret = None
validate = False
verbose = False
unsupervised = False
print_all = False
turbo = False
selected_datasets = list( map( str, range( 1, 11 ) ) )
selected_models = None
excluded_models = None

for key, value in options:
	match key:
		case "-t":
			train = True
		case "-v":
			validate = True
		case "-p":
			pycaret = value
		case "-u":
			unsupervised = True
		case "-d":
			selected_datasets = list( map( str.strip, value.split( "," ) ) )
		case "-o":
			oversampling_threshold = int( value )
		case "-a":
			print_all = True
		case "-e":
			verbose = True
		case "-r":
			turbo = True
		case "-m":
			selected_models = list( map( str.strip, value.split( "," ) ) )
		case "-x":
			excluded_models = list( map( str.strip, value.split( "," ) ) )

with open( get_file_path( "metadata.json" ) ) as metadata_file:
	metadata = json.load( metadata_file )
	metadata_file.close()

available_models = get_metadata_setting( "supervised_algorithms", {} )

if selected_models is None:
	selected_models = list( available_models.keys() )
else:
	selected_models = expand_model_names( selected_models )

if excluded_models:
	excluded_models = expand_model_names( excluded_models )

	for model in excluded_models:
		if model in selected_models:
			selected_models.remove( model )

for dataset_id in selected_datasets:
	dataset = None
	file_path = None
	extension = None

	print_message( f"Dataset {dataset_id}..." )

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
		if not train or not verbose:
			print_message( "Preprocessing..." )

		preprocessed_dataset = dataset.copy()

		rows_to_remove = []

		preprocessed_dataset = preprocessed_dataset.rename( columns = lambda x: x.strip() )

		for col in preprocessed_dataset.columns:
			if get_column_metadata( dataset_id, col, "allow_na" ):
				preprocessed_dataset[ col ] = preprocessed_dataset[ col ].fillna( '' )
			else:
				preprocessed_dataset[ col ] = preprocessed_dataset[ col ].apply( lambda x : x.strip() if isinstance( x, str ) else x ).replace( "", np.nan )

		for index in preprocessed_dataset.index:
			if ( preprocessed_dataset.iloc[ index ].isna().sum() / len( preprocessed_dataset.iloc[ index ] ) ) > row_acceptance_threshold:
				rows_to_remove.append( index )

		preprocessed_dataset = preprocessed_dataset.drop( index = rows_to_remove )

		columns_to_remove = []

		for col in preprocessed_dataset.columns:
			if ( preprocessed_dataset[ col ].isnull().sum() / len( preprocessed_dataset[ col ] ) ) > column_acceptance_threshold:
				columns_to_remove.append( col )

		preprocessed_dataset = (
			preprocessed_dataset
				.drop( columns = columns_to_remove )
				.dropna()
		)

		if not train or not verbose:
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

	# Apply Minmaxscaler
	scaler = MinMaxScaler()
	engineered_dataset[ engineered_dataset.columns ] = scaler.fit_transform( engineered_dataset[ engineered_dataset.columns ] )

	# Correlation analysis
	correlation_matrix = engineered_dataset.corr().abs()

	upper = correlation_matrix.where( np.triu( np.ones( correlation_matrix.shape ), k = 1).astype( bool ) )

	to_drop = set()

	target_column = get_target_column( dataset_id )

	for col1 in upper.columns:
		if col1 != target_column:
			for col2 in upper.index:
				if col2 != target_column:
					if upper.loc[ col2, col1 ] > correlation_acceptance_threshold:
						# Determine which column to drop: the one with the lower relationship with the target column
						relationship_with_target = {
							col1: upper.loc[ col1, target_column ],
						  	col2: upper.loc[ col2, target_column ]
						}
						col_to_drop = min( relationship_with_target, key = relationship_with_target.get )
						to_drop.add( col_to_drop )

	engineered_dataset = engineered_dataset.drop( columns = list( to_drop ) )

	if not train or not verbose:
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

		if not train or not verbose:
			print_message( f"One-hot encoding on dataset #{dataset_id}, {len(categorical_cols_to_encode)} column/s: {message}" )

		engineered_dataset = pd.get_dummies( engineered_dataset, columns = columns, dummy_na = False ) # dummy_na=False to not create a column for NaN

	# Drop duplicate rows
	engineered_dataset = engineered_dataset.drop_duplicates()

	if not train or not verbose:
		print_message( f"Dataset #{dataset_id} dimension modification after engineering from {preprocessed_dataset.shape} to {engineered_dataset.shape}.")

	column_renamer = lambda x: re.sub( "[^A-Za-z0-9_]+", "_", x )
	engineered_dataset = engineered_dataset.rename( columns = column_renamer )
	unprocessed_dataset = unprocessed_dataset.rename( columns = column_renamer )

	if validate:
		# Separating for validation
		if target_column is not None:
			target_column = column_renamer( target_column )

			X = engineered_dataset.drop( target_column, axis = 1 )
			y = engineered_dataset[ target_column ]

			X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = validation_ratio, random_state = random_seed )

			engineered_dataset = pd.concat( [ X_train, pd.Series( y_train, name = target_column ) ], axis = 1 )
			validation_dataset = pd.concat( [ X_validation, pd.Series( y_validation, name = target_column ) ], axis = 1 )
		else:
			engineered_dataset, validation_dataset = train_test_split( engineered_dataset, test_size = validation_ratio, random_state = random_seed )

		if not train or not verbose:
			print_message( f"Engineered dataset: {engineered_dataset.shape}, Validation dataset: {validation_dataset.shape}" )

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

		balanced_dataset = pd.concat( [ X, pd.Series( y, name = target_column ) ], axis = 1 )

		if not train or not verbose:
			print_message( f"Dataset #{dataset_id} dimension modification after balancing from {engineered_dataset.shape} to {balanced_dataset.shape}.")

	if train:
		if target_column is None:
			print_message( "No target column. Skipping supervised analysis." )
			pass
		else:
			match pycaret:
				case 'only_training':
					experiments_settings = {
						"Pycaret only training" : {
							"dataset" : balanced_dataset,
							"pycaret_setup_args" : {
								"preprocess" : False
							}
						},
						"Pycaret only training and tuning" : {
							"dataset" : balanced_dataset,
							"tune" : True,
							"pycaret_setup_args" : {
								"preprocess" : False
							}
						}
					}
				case 'complete':
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
							"dataset" : balanced_dataset,
						},
						"Custom with tuning" : {
							"dataset" : balanced_dataset,
							"tune" : True,
						},
					}

			for key, value in experiments_settings.items():
				if experiment( key, value.get( "dataset" ), tune = value.get( "tune", False ), validation_dataset = validation_dataset if validate else None, **value.get( "pycaret_setup_args", {} ) ):
					break
