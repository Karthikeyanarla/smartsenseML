#Load the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load the data
df = pd.read_csv('./cardata.csv')

# Checking for the duplicate data
df.duplicated().sum()

# using one hot encoding to convert the categorical data.
one_hot_data = pd.get_dummies(df[['State', 'Make', 'Model']])
df.drop(columns=['State', 'Make', 'Model'], inplace= True)
final_data = pd.concat([one_hot_data, df], axis='columns')
df = final_data


# df contains the X and y, spliting that
X = df.drop(['Price'], axis=1).reset_index(drop=True)
y = df['Price']


# split the data into test and training 75 percent training and 25 percent test data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


from tqdm import tqdm

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

regr_1 = DecisionTreeRegressor(max_depth=40)

regr_1.fit(X_train, y_train)
y_1 = regr_1.predict(X_test)


# use mse to calculate the error
from sklearn.metrics import mean_squared_error
print("mse, ", mean_squared_error(y_test, y_1))

import matplotlib.pyplot as plt
# give colors to the graphs
plt.figure(figsize=(20,10))
plt.plot(y_1[:50], c = 'b')
plt.plot(y_test[:50], c = 'r')

# give legend to the graph
plt.title("Prediction vs Actual")
plt.xlabel("No of samples")
plt.ylabel("Price")
plt.legend(["Actual", "Predicted"], loc ="upper left")
plt.savefig("images/prediction_actual_graph.png")
plt.show()
plt.close()

from sklearn.model_selection import cross_val_score
dt_cv_scores = cross_val_score(regr_1, X, y, cv=10)
print('Cross-validation scores: ', dt_cv_scores)
print('Mean CV score: ', np.mean(dt_cv_scores))

# save the predicted values and actuals in a csv file
y_1 = pd.DataFrame(y_1)
y_test = pd.DataFrame(y_test)
pd.concat(y_1, y_test)
y_1.to_csv('predicted and actual.csv')
