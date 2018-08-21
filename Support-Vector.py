import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', 1000)


iris = sns.load_dataset('iris')

print(iris.head())
print(iris.info())
print(iris.describe())

sns.pairplot(iris, hue='species')
plt.show()

setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'],cmap='plasma',shade=True, shade_lowest=False)
#sns.kdeplot(iris['sepal_length'], iris['sepal_width'], shade=True)
plt.show()



##################  TRAINING AND TESTING  ######################

from sklearn.model_selection import train_test_split

#X = pd.DataFrame(iris, columns=iris.columns[:-1])
X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


from sklearn.grid_search import GridSearchCV

param_grid = {'C':[.1,1,10,100,1000], 'gamma': [1,.1,.01,.001,.0001,.00001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

print(grid.best_params_)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))

