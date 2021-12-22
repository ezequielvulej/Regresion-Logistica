# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 08:52:07 2021

@author: evule
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Datos
sexo=['Hombre','Hombre','Mujer','Otro','Hombre','Hombre','Mujer','Mujer','Otro','Mujer']
edad=[41,50,22,19,35,61,28,28,39,20]
deudaDefaulteada=[1000,900,0,0,0,1800,100,0,0,0]

#Manipulacion de datos:
matrizSexo=pd.get_dummies(sexo)

def grupoEdad(x):
    for i in range(10):
        if i*10<=x<(i+1)*10:
            grupo="De "+ str(i*10) +" a " + str((i+1)*10) +" años"
        elif x>100:
            grupo="Mayor que 100"
    return grupo

matrizEdad=pd.get_dummies(list(map(grupoEdad,edad)))

X=pd.concat([matrizSexo,matrizEdad], axis=1)
y=[1 if i>0 else 0 for i in deudaDefaulteada]

#Separacion de datos:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Definicion y entrenamiento del modelo
model=LogisticRegression(solver="liblinear", random_state=0, penalty="l2", C=3.0)
model.fit(X_train,y_train)

#Evaluacion del modelo:
probabilidades=model.predict_proba(X_train) #La primera columna es la probabilidad de predecir 0. La segunda, de predecir 1.
print(probabilidades)
predicciones=model.predict(X_train) #Las predicciones (0 si la probabilidad de 0 es menor que 0.5)
print(predicciones)


print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))




