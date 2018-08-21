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

sns.kdeplot(iris['sepal_length'], iris['sepal_width'], shade=True)
plt.show()



##################  TRAINING AND TESTING  ######################

#from sklearn.mod