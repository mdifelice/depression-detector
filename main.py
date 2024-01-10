from classes.depressiondetector import DepressionDetector
from classes.trainingdataset import TrainingDataset
import os

os.environ["KAGGLE_USERNAME"] = "martindanieldifelice"
os.environ["KAGGLE_KEY"] = "55268f1d3587d8139d0404ee1d48f3ea"

dp = DepressionDetector()

td = TrainingDataset(
	"https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data",
	"depression_diagnosis",
	dataset_file = "depression_anxiety_data.csv"
)

dp.add_training_dataset( td )

dp.train( tune = True )
