from classes.depressiondetector import DepressionDetector
from classes.dataset import Dataset
import os

os.environ["KAGGLE_USERNAME"] = "martindanieldifelice"
os.environ["KAGGLE_KEY"] = "55268f1d3587d8139d0404ee1d48f3ea"

dp = DepressionDetector()

d = Dataset(
	"https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data",
	"depression_diagnosis",
	file = "depression_anxiety_data.csv"
)

dp.add_training_dataset( d )

dp.train( tune = True )
