from .depressiondetector import DepressionDetector

dataset = DepressionDetector.load_dataset_from_kaggle( "shahzadahmad0402/depression-and-anxiety-data", "depression_anxiety_data.csv" )

DepressionDetector.train( dataset )
