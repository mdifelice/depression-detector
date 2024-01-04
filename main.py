from classes.depressiondetector import DepressionDetector
import os

os.environ["KAGGLE_USERNAME"] = "martindanieldifelice"
os.environ["KAGGLE_KEY"] = "55268f1d3587d8139d0404ee1d48f3ea"

dp = DepressionDetector()

dataset = dp.load_dataset_from_kaggle( "shahzadahmad0402/depression-and-anxiety-data", "depression_anxiety_data.csv", "./datasets" )

dp.train( dataset, 'depression_diagnosis' )
