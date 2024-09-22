
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# TAKING CARE OF MISSING DATA
missing_data = dataset.isnull().sum()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# GETTING THE MISSING DATA COLUMN
imputer.fit(x[:, 1:3])
# REPLACING THE MISSING DATA
x[:, 1:3] = imputer.transform(x[:, 1:3])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')


x = np.array(ct.fit_transform(x))
#Encoding the data
le = LabelEncoder()
y = le.fit_transform(y)
#splitting the dataset into the training set and the test set
x_train,x_test , y_train , y_test = train_test_split( x, y, test_size=0.2, random_state=1 )

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

# print(x_train)
print(x_test)

# print("Missing data: \n", missing_data)
