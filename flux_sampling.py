from cobra.io import read_sbml_model
from cobra.sampling import sample
import pandas as pd

lung_model = read_sbml_model("lung_macrophages.xml")
s = sample(lung_model, 10000)
s.to_csv('lung_10k.csv')

liver_model = read_sbml_model("liver_hepatocytes.xml")
s = sample(liver_model, 10000)
s.to_csv('liver_10k.csv')

muscle_model = read_sbml_model("skeletal_muscle_myocytes.xml")
s = sample(muscle_model, 10000)
s.to_csv('muscle_10k.csv')