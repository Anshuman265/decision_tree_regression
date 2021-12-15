#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the decision Tree regression on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X,y)

# No need to apply feature scaling in decision tree regressions , because they are not equation

# Predicting the result
regressor.predict([[6.5]])

#Visualising the decision tree regression (higher resolution)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

