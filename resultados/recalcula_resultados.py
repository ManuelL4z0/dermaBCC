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
path_ = 'predicciones/3Entradas_ReticulosCorregidos/RGB'
path_2= 'prueba bucle'
_predicciones = '_predicciones.npy'
_etiquetas    = '_etiquetasY.npy'

predicciones = np.load(os.path.join(path_,_predicciones))
etiquetas = np.load(os.path.join(path_,_etiquetas))
umbrales = np.linspace(0,1,num=20)
valores = []
for umbral in umbrales:
    print(umbral)
    test_pred = predict_triple2(predicciones,umbral)
    cm = confusion_matriz(test_pred,etiquetas)
    valores.append(calculo_valores(cm))
valores = np.asarray(valores)
print("Umbral - sensibilidad - espeficicidad")
# for j,umbral in enumerate(umbrales):
#     print(umbral,sum(valores[j,3,:])/7)
#     print(sum(valores[j,2,:])/7)
#%%
for i in range(7):
    print("patron: ",i)
    for j in range(len(umbrales)):
        media = np.sum(valores[j,2:,i])/2
        print(media,j,valores[j,2:,i])

    # for j,umbral in enumerate(umbrales):
    #     print(umbral,sum(valores[j,3,:])/7)
    #     print(sum(valores[j,2,:])/7)
# medias = sum(valores[:,3,:])/7
# for i in range(len(predicciones))
#%%
vectorUmbrales = [umbrales[16],umbrales[4],umbrales[1],umbrales[1],
                  umbrales[5],umbrales[1],umbrales[4]]
test_pred = predict_triple3(predicciones,vectorUmbrales)
cm = confusion_matriz(test_pred,etiquetas)
calculo_valores(cm)