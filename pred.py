# library import
import pandas as pd
import tensorflow as tf

# past data preperation
file1 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(file1)
print(boston.columns)
boston.head()

# distinguish feature, target
feature = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
target = boston[['medv']]

# model makeing
X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X)
y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X,y)
model.compile(loss='mse')

#H = tf.keras.layers.Dense(4, activation='swish')(X)
#H1 = tf.keras.layers.Dense(4, activation='swish')(H)
#y = tf.keras.layers.Dense(1)(H1)

# model learning
model.fit(feature,target,epochs=1000,verbose=0)

model.predict(feature)
