from .util import get_data_folder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from urllib.parse import urlparse
from zipfile import ZipFile
import kaggle
import numpy as np
import os
import pandas as pd
import re

class Dataset:
	def __init__( self, url, target_column, file = None ):
		parsed_url = urlparse( url )

		if parsed_url.hostname == "www.kaggle.com":
			id = re.sub( r"^\/datasets\/(.+)\/?$", r"\1", parsed_url.path )
			basename = id.split( "/" )[-1]
			folder = get_data_folder( "datasets" ) + basename
			zip = folder + "/" + basename + ".zip"

			kaggle.api.dataset_download_files( id, folder )

			with ZipFile( zip , "r") as zip_object:
				zip_object.extractall( folder )

			os.remove( zip )

			if file:
				file = folder + "/" + file

		if file:
			extension = file.split( "." )[-1]

			if "csv" == extension:
				callback = pd.read_csv
			elif "xlsx" == extension:
				callback = pd.read_excel

		if callback and file:
			data = callback( file )
		else:
			raise Exception( "Invalid dataset" )

		self.target_column = target_column
		self.versions = {}
		self.versions["original"] = data

	def train( tune = False ):
		pass

	def __prepare_data( self ):
		categorical_columns = []
		numerical_columns = []
		unique_columns = []

		debug( "Analyzing features..." )

		for column_name in self.versions["original"].columns:
			column = self.versions["original"][ column_name ].dropna()

			if pd.api.types.is_numeric_dtype( column.dtype ):
				numerical_columns.append( column_name )

				if column.is_unique:
					unique_columns.append( column_name )
			elif column.dtype == "object":
				uniques = column.unique()

				uniques.sort()

				if len( uniques ) == 2 and uniques[0] == False and uniques[1] == True:
					self.versions["original"][ column_name ] = column.astype( int )
				else:
					categorical_columns.append( column_name )

		debug( "Normalizing versions..." )

		standard_scaler = StandardScaler()
		min_max_scaler = MinMaxScaler()

		self.versions["original_standard_scaled"] = pd.DataFrame( self.versions["original"], columns = self.versions["original"].columns )
		self.versions["original_standard_scaled"][ numerical_columns ] = standard_scaler.fit_transform( self.versions["original_standard_scaled"][ numerical_columns ] )

		self.versions["original_min_max_scaled"] = pd.DataFrame( self.versions["original"], columns = self.versions["original"].columns )
		self.versions["original_min_max_scaled"][ numerical_columns ] = min_max_scaler.fit_transform( self.versions["original_min_max_scaled"][ numerical_columns ] )

		if len( categorical_columns ):
			debug( "Generating one-hot versions..." )

			self.versions["onehot"] = pd.get_dummies( self.versions["original"], columns = categorical_columns )
			self.versions["onehot_standard_scaled"] = pd.get_dummies( self.versions["original_standard_scaled"], columns = categorical_columns )
			self.versions["onehot_min_max_scaled"] = pd.get_dummies( self.versions["original_min_max_scaled"], columns = categorical_columns )

		self.__debug( "Cleaning versions..." )

		for id in self.versions:
			debug( "Cleaning version " + id + "..." )

			data = self.versions[ id ]

			debug( "Removing duplicates..." )

			data.drop_duplicates( inplace = True )

			debug( "Removing null values..." )

			data.dropna( inplace = True )
	
		debug( "Creating feature-optimized versions..." )

		custom_limits = [ 0.4, 0.6, 0.8 ]

		# We only use SelectKBest over one hot versions since the algorithm does not work with string values
		select_k_best_estimators = {
			"chi2" : {
				"estimator" : chi2,
				"validator" : lambda df : ( df.values >= 0 ).all(),
			},
			"f_classif" : {
				"estimator" : f_classif,
			}
		}

		ids = self.versions.copy().keys()

		for id in ids:
			debug( "Working with version " + id + "..." )

			for limit in custom_limits:
				debug( "Applying feature reduction with limit " + str( limit ) + "..." )

				self.versions[ id + "_reduced_custom_" + str( limit ) ] = self.__reduce_features( self.versions[ id ], limit, target_column )

			if re.search( r"^onehot_", id ):
				for name, settings in select_k_best_estimators.items():
					debug( "Applying select k-best using " + name + "..." )

					df = self.versions[ id ]

					validator = settings.get( "validator" )

					if not validator or validator( df ):
						estimator =  SelectKBest( settings.get( "estimator" ) )

						estimator.fit_transform( df.drop( target_column, axis = 1 ), df[ target_column ] )

						features = estimator.get_feature_names_out()

						self.versions[ id + "_reduced_" + name ] = self.versions[ id ].loc[:, np.append( features, [ target_column ] ) ]

	def __reduce_features( self, df, limit, target_column ):
		df_reduced = df.copy()

		while True:
			columns_to_remove = []

			correlation = df_reduced.corr( numeric_only = True )
			correlation_keys = correlation.keys()

			for column in correlation:
				for i in range( correlation_keys.get_loc( column ) + 1, len( correlation_keys ) ):
					column_to_remove = correlation.columns[ i ]

					if correlation[column][ column_to_remove ] >= limit:
						if column_to_remove != target_column and column_to_remove not in columns_to_remove:
							columns_to_remove.append( column_to_remove )

			if not len( columns_to_remove ):
				break
			else:
				df_reduced.drop( columns = columns_to_remove, inplace = True )

		return df_reduced
