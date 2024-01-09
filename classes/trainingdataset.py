class TrainingDataset:
	def __init__( self, url, target_column ):
#def load_dataset_from_kaggle( self, dataset_id, dataset_file, base_folder ):
# todo parse url from kaggle
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
			data = callback( folder + '/' + dataset_file )
		else:
			raise Exception( 'Invalid dataset' )

		self.target_column = target_column
		self.data = {}
		self.data['original'] = data

	def __prepare_data( self ):
		categorical_columns = []
		numerical_columns = []
		unique_columns = []

		debug( 'Analyzing features...' )

		for column_name in self.data['original'].columns:
			column = self.data['original'][ column_name ].dropna()

			if pd.api.types.is_numeric_dtype( column.dtype ):
				numerical_columns.append( column_name )

				if column.is_unique:
					unique_columns.append( column_name )
			elif column.dtype == 'object':
				uniques = column.unique()

				uniques.sort()

				if len( uniques ) == 2 and uniques[0] == False and uniques[1] == True:
					self.data['original'][ column_name ] = column.astype( int )
				else:
					categorical_columns.append( column_name )

		debug( 'Normalizing datasets...' )

		standard_scaler = StandardScaler()
		min_max_scaler = MinMaxScaler()

		self.data['original_standard_scaled'] = pd.DataFrame( self.data['original'], columns = self.data['original'].columns )
		self.data['original_standard_scaled'][ numerical_columns ] = standard_scaler.fit_transform( self.data['original_standard_scaled'][ numerical_columns ] )

		self.data['original_min_max_scaled'] = pd.DataFrame( self.data['original'], columns = self.data['original'].columns )
		self.data['original_min_max_scaled'][ numerical_columns ] = min_max_scaler.fit_transform( self.data['original_min_max_scaled'][ numerical_columns ] )

		if len( categorical_columns ):
			debug( 'Generating one-hot datasets...' )

			self.data['onehot'] = pd.get_dummies( self.data['original'], columns = categorical_columns )
			self.data['onehot_standard_scaled'] = pd.get_dummies( self.data['original_standard_scaled'], columns = categorical_columns )
			self.data['onehot_min_max_scaled'] = pd.get_dummies( self.data['original_min_max_scaled'], columns = categorical_columns )

		self.__debug( 'Cleaning data...' )

		for dataset_id in self.data:
			debug( 'Cleaning version ' + dataset_id + '...' )

			data = self.data[ dataset_id ]

			debug( 'Removing duplicates...' )

			data.drop_duplicates( inplace = True )

			debug( 'Removing null values...' )

			data.dropna( inplace = True )
	
		debug( 'Creating feature-optimized versions...' )

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

		dataset_ids = self.data.copy().keys()

		for dataset_id in dataset_ids:
			debug( 'Working with dataset ' + dataset_id + '...' )

			for limit in custom_limits:
				debug( 'Applying feature reduction with limit ' + str( limit ) + '...' )

				self.data[ dataset_id + '_reduced_custom_' + str( limit ) ] = self.__reduce_features( self.data[ dataset_id ], limit, target_column )

			if re.search( r"^onehot_", dataset_id ):
				for name, settings in select_k_best_estimators.items():
					debug( 'Applying select k-best using ' + name + '...' )

					dataset = self.data[ dataset_id ]

					validator = settings.get( 'validator' )

					if not validator or validator( dataset ):
						estimator =  SelectKBest( settings.get( 'estimator' ) )

						estimator.fit_transform( dataset.drop( target_column, axis = 1 ), dataset[ target_column ] )

						features = estimator.get_feature_names_out()

						self.data[ dataset_id + '_reduced_' + name ] = self.data[ dataset_id ].loc[:, np.append( features, [ target_column ] ) ]

	def __reduce_features( self, dataset, limit, target_column ):
		dataset_reduced = dataset.copy()

		while True:
			columns_to_remove = []

			correlation = dataset_reduced.corr( numeric_only = True )
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
				dataset_reduced.drop( columns = columns_to_remove, inplace = True )

		return dataset_reduced
