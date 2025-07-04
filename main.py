#!.venv/bin/python
import os
import json
import pandas as pd
import numpy as np
import re
import sys
from classes.util import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from pycaret.classification import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from getopt import getopt

def get_column_metadata( dataset_id, col, field ):
	value = None

	if metadata:
		if dataset_id in metadata:
			if col in metadata[ dataset_id ]:
				value = metadata[ dataset_id ].get( col ).get( field )

	return value

def get_target_column( dataset_id ):
	target_column = None

	for col in dataset.columns:
		if get_column_metadata( dataset_id, col, "target" ):
			target_column = col
			break

	return target_column

def get_formatted_metrics( success, metrics ):
	return " \033[" + ( "32" if success else "31" ) + "m" + f"F1 {metrics["F1"]}, AUC {metrics["AUC"]}, Accuracy {metrics["Accuracy"]}, Precision {metrics["Prec."]}, Recall {metrics["Recall"]}" + "\033[0m"

def experiment( title, dataset, tune = False, validation_dataset = None, **pycaret_setup_args ):
	debug( title + "...", eol = False )

	original_stdout = sys.stdout
	sys.stdout = open( os.devnull, "w" )

	setup(
		data = dataset,
		target = target_column,
		session_id = random_seed,
		train_size = 1 - test_ratio,
		verbose = False,
		**pycaret_setup_args
	)

	compare_models(
		sort = "f1",
#		errors = "raise",
		verbose = False
	)

	sys.stdout = original_stdout

	metrics = pull()

	match = None
	model = None
	success = False
	message = "No metrics found."
	len_metrics = len( metrics )

	if len_metrics:
		max_auc = 0
		match_index = 0

		for index in range( len_metrics ):
			row = metrics.iloc[ index ]

			if row["F1"] >= f1_acceptance_threshold:
				if row["AUC"] > max_auc:
					max_auc = row["AUC"]
					match_index = index

				success = True
			else:
				break

		match = metrics.iloc[ match_index ]
		model_id = metrics.index[ match_index ]
		model = create_model(
			model_id,
			verbose = False
		)

		if tune:
			tuned_model = tune_model(
				model,
				optimize = 'F1',
				n_iter = 100,
				verbose = False
			)

			metrics = pull()

			match = metrics.loc['Mean']

			success = match["F1"] > f1_acceptance_threshold

		message = f" {model_id}: {get_formatted_metrics( success, match )}"

	debug( message, timestamp = False )

	if success and model is not None and validation_dataset is not None:
		predictions = predict_model( model, data = validation_dataset, verbose = False )

		metrics = pull()

		debug( f"Validation: {get_formatted_metrics( metrics.loc[0]["F1"] > f1_acceptance_threshold, metrics.loc[0] )}")
	
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

try:
	options, extra_args = getopt( sys.argv[ 1: ], "d:tpv" )
except Exception as e:
	print( f"Error parsing arguments: {e}" )
	sys.exit(1)

train = False
pycaret = False
validate = False
selected_datasets = list( map( str, range( 1, 11 ) ) )

for key, value in options:
	match key:
		case "-t":
			train = True
		case "-v":
			validate = True
		case "-p":
			pycaret = True
		case "-d":
			selected_datasets = list( map( str.strip, value.split( "," ) ) )

with open( get_file_path( "metadata.json" ) ) as metadata_file:
	metadata = json.load( metadata_file )
	metadata_file.close()

for dataset_id in selected_datasets:
	dataset = None
	file_path = None
	extension = None

	debug( f"Dataset {dataset_id}..." )

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
		debug( "Preprocessing..." )
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

		debug( f"Dataset #{dataset_id} dimension reduction from {dataset.shape} to {preprocessed_dataset.shape}.")

		debug( "Engineering..." )
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

		dataset_for_pycaret = engineered_dataset.copy()

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

	debug( f"Correlation analysis on dataset #{dataset_id}, removing {len(to_drop)} column/s: {sorted(to_drop)}" )

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

		debug(f"One-hot encoding on dataset #{dataset_id}, {len(categorical_cols_to_encode)} column/s: {message}")

		engineered_dataset = pd.get_dummies( engineered_dataset, columns = columns, dummy_na = False ) # dummy_na=False to not create a column for NaN

	# Drop duplicate rows
	engineered_dataset = engineered_dataset.drop_duplicates()

	debug( f"Dataset #{dataset_id} dimension modification after engineering from {preprocessed_dataset.shape} to {engineered_dataset.shape}.")

	column_renamer = lambda x: re.sub( "[^A-Za-z0-9_]+", "_", x )
	engineered_dataset = engineered_dataset.rename( columns = column_renamer )
	dataset_for_pycaret = dataset_for_pycaret.rename( columns = column_renamer )

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

		debug( f"Engineered dataset: {engineered_dataset.shape}, Validation dataset: {validation_dataset.shape}" )

	if target_column is not None:
		target_column = column_renamer( target_column )

		# todo check Nan values
		X = engineered_dataset.drop( target_column, axis = 1 )
		y = engineered_dataset[ target_column ]

		oversampling = len(y) < 1000

		if oversampling:
			rs = SMOTE( random_state = random_seed )
		else:
			rs = RandomUnderSampler( random_state = random_seed )

		X, y = rs.fit_resample( X, y )

		balanced_dataset = pd.concat( [ X, pd.Series( y, name = target_column ) ], axis = 1 )

		debug( f"Dataset #{dataset_id} dimension modification after balancing from {engineered_dataset.shape} to {balanced_dataset.shape}.")

	if train:
		if target_column is None:
			debug( "No target column. Skipping supervised analysis." )
			pass
		else:
			experiments_settings = {
				"Custom" : {
					"dataset" : balanced_dataset,
					"pycaret_setup_args" : {
						"preprocess" : False
					}
				},
				"Custom (tuned)" : {
					"dataset" : balanced_dataset,
					"tune" : True,
					"pycaret_setup_args" : {
						"preprocess" : False
					}
				}
			}

			if pycaret:
				experiments_settings |= {
					"Pycaret" : {
						"dataset" : dataset_for_pycaret
					},
					"Pycaret (normalized)" : {
						"dataset" : dataset_for_pycaret,
						"pycaret_setup_args" : {
							"normalize" : True
						}
					},
					"Pycaret (balanced)" : {
						"dataset" : dataset_for_pycaret,
						"pycaret_setup_args" : {
							"fix_imbalance" : True
						}
					},
					"Pycaret (normalized/balanced)" : {
						"dataset" : dataset_for_pycaret,
						"pycaret_setup_args" : {
							"normalize" : True,
							"fix_imbalance" : True
						}
					},
					"Pycaret (normalized/balanced/tuned)" : {
						"dataset" : dataset_for_pycaret,
						"tune" : True,
						"pycaret_setup_args" : {
							"normalize" : True,
							"fix_imbalance" : True
						}
					}
				}

			for key, value in experiments_settings.items():
				if experiment( key, value.get( "dataset" ), tune = value.get( "tune", False ), validation_dataset = validation_dataset if validate else None, **value.get( "pycaret_setup_args", {} ) ):
					break
