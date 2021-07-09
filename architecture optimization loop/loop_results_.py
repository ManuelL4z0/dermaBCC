# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:25:24 2021

@author: Manuel
"""
import numpy as np
import os
from confusion_matrix_custom import confusion_matriz,calculo_valores,predict_triple2,predict_triple3
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#%%
path_= 'prueba_bucle'
listaCarpetas = os.listdir(path_)
#%%
_predicciones = '_predicciones.npy'
_etiquetas    = '_etiquetasY.npy'

for carpeta in listaCarpetas:
    path_1 = os.path.join(path_,carpeta)
    print(carpeta)
    if 'resultados.txt' not in os.listdir(path_1):
        print(carpeta)
        predicciones = np.load(os.path.join(path_1,_predicciones))
        etiquetas = np.load(os.path.join(path_1,_etiquetas))
        umbrales = np.linspace(0,1,num=20)
        valores = []
        for umbral in umbrales:
            # print(umbral)
            test_pred = predict_triple2(predicciones,umbral)
            cm = confusion_matriz(test_pred,etiquetas)
            valores.append(calculo_valores(cm))
        valores = np.asarray(valores)
        vectorUmbrales=np.zeros(7)
        for i in range(7):
            media = np.zeros(len(umbrales))
            # print("patron: ",i)
            for j in range(len(umbrales)):
                media[j] = np.sum(valores[j,2:,i])/2
                # print(media,j,valores[j,2:,i])
            vectorUmbrales[i] = umbrales[np.argmax(media)]
        # print(vectorUmbrales)
        test_pred = predict_triple3(predicciones,vectorUmbrales)
        cm = confusion_matriz(test_pred,etiquetas)
        val = calculo_valores(cm)
        np.savetxt(os.path.join(path_1,'resultados.txt'),val)
        # break
