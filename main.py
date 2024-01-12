from classes.depressiondetector import DepressionDetector
import os

os.environ["KAGGLE_USERNAME"] = "martindanieldifelice"
os.environ["KAGGLE_KEY"] = "55268f1d3587d8139d0404ee1d48f3ea"

dp = DepressionDetector()

dp.add_dataset(
	"https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data",
	"depression_diagnosis",
	file = "depression_anxiety_data.csv"
)

dp.train( tune = True )
