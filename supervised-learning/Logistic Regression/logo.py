import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection     import train_test_split
from sklearn.compose             import ColumnTransformer
from sklearn.preprocessing       import OneHotEncoder, StandardScaler
from imblearn.pipeline           import Pipeline as ImbPipeline
from imblearn.over_sampling      import SMOTE
from sklearn.linear_model        import LogisticRegression
from sklearn.metrics             import classification_report, confusion_matrix

# 1) Load the cleaned dataset
df = pd.read_csv("/Users/pbat/Projects/cmor438/data/Cleaned_dataset.csv")
