from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class DepressionDetector:
	def __init__( self ):
		pass

	def train( self, dataset ):
		datasets = {}

		# Read tmp?

		datasets['original'] = self.get_dataset()

		categorical_columns = []
		numerical_columns = []

		for column_name in datasets['original'].columns:
		  column = datasets['original'][ column_name ].dropna()

		  if pd.api.types.is_numeric_dtype( column.dtype ):
			numerical_columns.append( column_name )
		  elif column.dtype == 'object':
			uniques = column.unique()

			uniques.sort()

			if len( uniques ) == 2 and uniques[0] == False and uniques[1] == True:
				datasets['original'][ column_name ] = column.astype( int )
			else:
				categorical_columns.append( column_name )

		standard_scaler = StandardScaler()
		min_max_scaler = MinMaxScaler()

		datasets['original_standard_scaled'] = pd.DataFrame( datasets['original'], columns = datasets['original'].columns )
		datasets['original_standard_scaled'][ numerical_columns ] = standard_scaler.fit_transform( datasets['original_standard_scaled'][ numerical_columns ] )

		datasets['original_min_max_scaled'] = pd.DataFrame( datasets['original'], columns = datasets['original'].columns )
		datasets['original_min_max_scaled'][ numerical_columns ] = min_max_scaler.fit_transform( datasets['original_min_max_scaled'][ numerical_columns ] )

		if len( categorical_columns ):
			datasets['onehot'] = pd.get_dummies( datasets['original'], columns = categorical_columns )
			datasets['onehot_standard_scaled'] = pd.get_dummies( datasets['original_standard_scaled'], columns = categorical_columns )
			datasets['onehot_min_max_scaled'] = pd.get_dummies( datasets['original_min_max_scaled'], columns = categorical_columns )

		for dataset_id in datasets:
			dataset = datasets[ dataset_id ]

			# Remove duplicates
			dataset.drop_duplicates( inplace = True )

			# Remove nulls
			dataset.dropna( inplace = True )
	
		# Feature reduction

		# Save tmp

		# Separate validation dataset

		# Train datasets

		# Tune models

	# Other: charts, comparisons, predict
