from sklearn.preprocessing import StandardScaler, MinMaxScaler
from zipfile import ZipFile
import pandas as pd

class DepressionDetector:
	__datasets = {}

	def __init__( self ):
		pass

	def train( self, dataset ):
		__datasets = {}

		# Read tmp?

		__datasets['original'] = dataset

		categorical_columns = []
		numerical_columns = []

		for column_name in __datasets['original'].columns:
		  column = __datasets['original'][ column_name ].dropna()

		  if pd.api.types.is_numeric_dtype( column.dtype ):
			numerical_columns.append( column_name )
		  elif column.dtype == 'object':
			uniques = column.unique()

			uniques.sort()

			if len( uniques ) == 2 and uniques[0] == False and uniques[1] == True:
				__datasets['original'][ column_name ] = column.astype( int )
			else:
				categorical_columns.append( column_name )

		standard_scaler = StandardScaler()
		min_max_scaler = MinMaxScaler()

		__datasets['original_standard_scaled'] = pd.DataFrame( __datasets['original'], columns = __datasets['original'].columns )
		__datasets['original_standard_scaled'][ numerical_columns ] = standard_scaler.fit_transform( __datasets['original_standard_scaled'][ numerical_columns ] )

		__datasets['original_min_max_scaled'] = pd.DataFrame( __datasets['original'], columns = __datasets['original'].columns )
		__datasets['original_min_max_scaled'][ numerical_columns ] = min_max_scaler.fit_transform( __datasets['original_min_max_scaled'][ numerical_columns ] )

		if len( categorical_columns ):
			__datasets['onehot'] = pd.get_dummies( __datasets['original'], columns = categorical_columns )
			__datasets['onehot_standard_scaled'] = pd.get_dummies( __datasets['original_standard_scaled'], columns = categorical_columns )
			__datasets['onehot_min_max_scaled'] = pd.get_dummies( __datasets['original_min_max_scaled'], columns = categorical_columns )

		for dataset_id in __datasets:
			dataset = __datasets[ dataset_id ]

			# Remove duplicates
			dataset.drop_duplicates( inplace = True )

			# Remove nulls
			dataset.dropna( inplace = True )
	
		print( __datasets )
		# Feature reduction

		# Save tmp

		# Separate validation dataset

		# Train datasets

		# Tune models

	# Other: charts, comparisons, predict

	def load_dataset_from_kaggle( dataset_id, dataset_file ):
		basename = dataset_id.split( '/' )[-1]
		folder = '/content/kaggle/' + basename
		zip = folder + '/' + basename + '.zip'

		kaggle.api.dataset_download_files( dataset_id, folder )

		with ZipFile( zip , 'r') as zip_object:
			zip_object.extractall( folder )

		os.remove( zip )

		extension = dataset_main_file.split( '.' )[-1]

		if 'csv' == extension:
			callback = pd.read_csv
		elif 'xlsx' == extension:
			callback = pd.read_excel
		else:
			callback = None

		if callback:
			dataset = callback( folder + '/' + dataset_main_file )
		else:
			dataset = None

		return dataset
