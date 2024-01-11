from .util import debug

class DepressionDetector:
	def __init__( self, random_seed = 1, validation_ratio = 0.2, train_sizes = [ 0.7, 0.8 ] ):
		self.__random_seed = random_seed
		self.__validation_ratio = validation_ratio
		self.__train_sizes = train_sizes
		self.__datasets = []

	def train( self, tune = False ):
		# Separate validation dataset
		debug( "Splitting train/test and validation data..." )

		validation_datasets = {};
		train_test_datasets = {};

		for dataset in self.__datasets:
			dataset = self.__datasets[ dataset_id ].sample( frac = 1, random_state = self.__random_seed )

			validation_limit = int( dataset.shape[0] * ( 1 - self.__validation_ratio ) )

			train_test_dataset = dataset[ 0:validation_limit ]
			validation_dataset = dataset[ validation_limit: ]

			train_test_datasets[ dataset_id ] = train_test_dataset
			validation_datasets[ dataset_id ] = validation_dataset

		# Train datasets
		debug( "Starting trainings..." )

		metrics = {} # Load tmp data
		models = {} # Load tmp data
		index = 0

#models:
# Logistic Regression
# Ridge Classifier
# Linear Discriminant Analysis
# Random Forest Classifier
# Naive Bayes
# CatBoost Classifier
# Gradient Boosting Classifier
# Ada Boost Classifier
# Extra Trees Classifier
# Quadratic Discriminant Analysis
# Light Gradient Boosting Machine
# K Neighbors Classifier
# Decision Tree Classifier
# Extreme Gradient Boosting
# Support Vector Machines
# Neural Networks ?
		for dataset_id in train_test_datasets:
			debug( "Trainings for dataset " + dataset_id + "..." )

			if not dataset_id in metrics:
				metrics[ dataset_id ] = {}

			if not dataset_id in models:
				models[ dataset_id ] = {}

			for train_size in self.__train_sizes:
				train_size_string = str( train_size )

				index += 1

				debug( "Using train size " + train_size_string + "..." )

				if not train_size_string in metrics[ dataset_id ]:
					pass
					# custom train
					# save models

		# Tune models
		if tune:
			debug( "Tuning models..." )
	# Other: charts, comparisons, predict

	def add_training_dataset( self, dataset, **kwargs ):
		self.__datasets.append( dataset, **kwargs )

	# to-do
	def predict( self ):
		pass
