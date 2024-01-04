from datetime import datetime
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from zipfile import ZipFile
import kaggle
import numpy as np
import os
import pandas as pd
import re

class DepressionDetector:
	def __init__( self ):
		self.random_seed = 123
		self.validation_ratio = 0.2

	def train( self, dataset, target ):
		self.__datasets = {}

		# Read tmp?

		self.__datasets['original'] = dataset

		categorical_columns = []
		numerical_columns = []
		unique_columns = []

		self.__debug( 'Analyzing features...' )

		for column_name in self.__datasets['original'].columns:
			column = self.__datasets['original'][ column_name ].dropna()

			if pd.api.types.is_numeric_dtype( column.dtype ):
				numerical_columns.append( column_name )

				if column.is_unique:
					unique_columns.append( column_name )
			elif column.dtype == 'object':
				uniques = column.unique()

				uniques.sort()

				if len( uniques ) == 2 and uniques[0] == False and uniques[1] == True:
					self.__datasets['original'][ column_name ] = column.astype( int )
				else:
					categorical_columns.append( column_name )

#if len( unique_columns ):
#			self.__debug( 'Removing unique columns...' )

#			dataset.drop( columns = unique_columns, inplace = True )

		self.__debug( 'Normalizing datasets...' )

		standard_scaler = StandardScaler()
		min_max_scaler = MinMaxScaler()

		self.__datasets['original_standard_scaled'] = pd.DataFrame( self.__datasets['original'], columns = self.__datasets['original'].columns )
		self.__datasets['original_standard_scaled'][ numerical_columns ] = standard_scaler.fit_transform( self.__datasets['original_standard_scaled'][ numerical_columns ] )

		self.__datasets['original_min_max_scaled'] = pd.DataFrame( self.__datasets['original'], columns = self.__datasets['original'].columns )
		self.__datasets['original_min_max_scaled'][ numerical_columns ] = min_max_scaler.fit_transform( self.__datasets['original_min_max_scaled'][ numerical_columns ] )

		if len( categorical_columns ):
			self.__debug( 'Generating one-hot datasets...' )

			self.__datasets['onehot'] = pd.get_dummies( self.__datasets['original'], columns = categorical_columns )
			self.__datasets['onehot_standard_scaled'] = pd.get_dummies( self.__datasets['original_standard_scaled'], columns = categorical_columns )
			self.__datasets['onehot_min_max_scaled'] = pd.get_dummies( self.__datasets['original_min_max_scaled'], columns = categorical_columns )

		self.__debug( 'Cleaning datasets...' )

		for dataset_id in self.__datasets:
			self.__debug( 'Cleaning dataset ' + dataset_id + '...' )

			dataset = self.__datasets[ dataset_id ]

			self.__debug( 'Removing duplicates...' )

			dataset.drop_duplicates( inplace = True )

			self.__debug( 'Removing null values...' )

			dataset.dropna( inplace = True )
	
		self.__debug( 'Creating feature-optimized datasets...' )

		custom_limits = [ 0.4, 0.6, 0.8 ]
		# We only use SelectKBest over one hot versions since the algorithm does not work with string values

		select_k_best_estimators = {
			'chi2' : {
				'estimator' : chi2,
				'validator' : lambda dataset : ( dataset.values >= 0 ).all(),
			},
			'f_classif' : {
				'estimator' : f_classif,
			}
		}

		dataset_ids = self.__datasets.copy().keys()

		for dataset_id in dataset_ids:
			self.__debug( 'Working with dataset ' + dataset_id + '...' )

			for limit in custom_limits:
				self.__debug( 'Applying feature reduction with limit ' + str( limit ) + '...' )

				self.__datasets[ dataset_id + '_reduced_custom_' + str( limit ) ] = self.__reduce_features( self.__datasets[ dataset_id ], limit, target )

			if re.search( r"^onehot_", dataset_id ):
				for name, settings in select_k_best_estimators.items():
					self.__debug( 'Applying select k-best using ' + name + '...' )

					dataset = self.__datasets[ dataset_id ]

					validator = settings.get( 'validator' )

					if not validator or validator( dataset ):
						estimator =  SelectKBest( settings.get( 'estimator' ) )

						estimator.fit_transform( dataset.drop( target, axis = 1 ), dataset[ target ] )

						features = estimator.get_feature_names_out()

						self.__datasets[ dataset_id + '_reduced_' + name ] = self.__datasets[ dataset_id ].loc[:, np.append( features, [ target ] ) ]

		# Save tmp

		# Separate validation dataset
		validation_datasets = {};
		train_test_datasets = {};

		for dataset_id in self.__datasets:
			dataset = self.__datasets[ dataset_id ].sample( frac = 1, random_state = self.random_seed )

			validation_limit = int( dataset.shape[0] * ( 1 - self.validation_ratio ) )

			train_test_dataset = dataset[ 0:validation_limit ]
			validation_dataset = dataset[ validation_limit: ]

			train_test_datasets[ dataset_id ] = train_test_dataset
			validation_datasets[ dataset_id ] = validation_dataset

		# Train datasets

		# Tune models

	# Other: charts, comparisons, predict

	def load_dataset_from_kaggle( self, dataset_id, dataset_file, base_folder ):
		basename = dataset_id.split( '/' )[-1]
		folder = base_folder + '/' + basename
		zip = folder + '/' + basename + '.zip'

		kaggle.api.dataset_download_files( dataset_id, folder )

		with ZipFile( zip , 'r') as zip_object:
			zip_object.extractall( folder )

		os.remove( zip )

		extension = dataset_file.split( '.' )[-1]

		if 'csv' == extension:
			callback = pd.read_csv
		elif 'xlsx' == extension:
			callback = pd.read_excel
		else:
			callback = None

		if callback:
			dataset = callback( folder + '/' + dataset_file )
		else:
			dataset = None

		return dataset
	
	def __reduce_features( self, dataset, limit, target ):
		dataset_reduced = dataset.copy()

		while True:
			columns_to_remove = []

			correlation = dataset_reduced.corr( numeric_only = True )
			correlation_keys = correlation.keys()

			for column in correlation:
				for i in range( correlation_keys.get_loc( column ) + 1, len( correlation_keys ) ):
					column_to_remove = correlation.columns[ i ]

					if correlation[column][ column_to_remove ] >= limit:
						if column_to_remove != target and column_to_remove not in columns_to_remove:
							columns_to_remove.append( column_to_remove )

			if not len( columns_to_remove ):
				break
			else:
				dataset_reduced.drop( columns = columns_to_remove, inplace = True )

		return dataset_reduced

	def __debug( self, message ):
		now = datetime.now()

		print( now.strftime( '[%Y-%m-%d %H:%M:%S] ' ) + message )
