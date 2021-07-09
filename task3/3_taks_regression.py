import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error


path = 'D:/Python/Quantum Test/Test Tesk_Internship/Quantum internship/'
data = pd.read_csv(path + 'internship_train.csv')

y = data.target
X = data.drop('target', axis=1) # cutof  column with target label


nb_degree = 2
polynomial_features = PolynomialFeatures(degree = nb_degree) 
X_TRANSF = polynomial_features.fit_transform(X) #change features for polinomial 

X_train, X_test, y_train, y_test = train_test_split(X_TRANSF, y, test_size=0.2)

model = LinearRegression() #define and train a model
model.fit(X_train, y_train)

#----------------------------------------------------------------------------------------#
Y_PRED = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,Y_PRED))

print('RMSE: ', rmse)

#----------------------------------------------------------------------------------------#
data_pr = pd.read_csv(path + 'internship_hidden_test.csv')

X_target_features = polynomial_features.fit_transform(data_pr)
y_target_pred = model.predict(X_target_features)

target_pred = np.asarray(y_target_pred)

plt.plot(target_pred,label='target_pred',color="red",alpha=0.5)
plt.title("taget prediction ")


plt.legend ()

plt.xlim(0,  200)
plt.show()  
data_pr

data_pr["target"] = target_pred
data_pr.to_csv(path+'internship_hidden_test_with_target_predicted.csv')


arr = np.asarray(y_test)
plt.plot(arr,label='y_test',color="red",alpha=0.5)
plt.plot( Y_PRED,label='Y_PRED',color="blue",alpha=0.5)
plt.title("taget test pred comparison ")
plt.legend ()
plt.xlim(0,  200)
plt.show()  
