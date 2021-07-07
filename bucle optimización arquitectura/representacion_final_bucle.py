# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:17:41 2021

@author: ManuelL
"""

import numpy as np
import os
from confusion_matrix_custom import confusion_matriz,calculo_valores,predict_triple2,predict_triple3
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#%%
path_= 'prueba_bucle'
listaCarpetas = os.listdir(path_)
print("Total de entrenamientos a revisar: ",len(listaCarpetas))
_predicciones = '_predicciones.npy'
_etiquetas    = '_etiquetasY.npy'
medias = np.zeros(len(listaCarpetas))
for n,carpeta in enumerate(listaCarpetas):
    path_1 = os.path.join(path_,carpeta)
    if 'resultados.txt' in os.listdir(path_1):
        valores = np.genfromtxt(os.path.join(path_,carpeta,'resultados.txt'),dtype=float)
        medias[n] = (sum(valores[2,:])/7+sum(valores[3,:])/7)/2
    # print(valores[0,1])
    # break
mediasOrdenadas = np.sort(medias)
indicesMediasOrdenadas = np.argsort(medias)

mejores10 =mediasOrdenadas[-30:]
mejores10Carpetas = np.array(listaCarpetas)[np.array(indicesMediasOrdenadas[-30:])]

peores10 =mediasOrdenadas[:30]
peores10Carpetas = np.array(listaCarpetas)[np.array(indicesMediasOrdenadas[:30])]

import matplotlib.pyplot as plt
fig1,ax1=plt.subplots(dpi=300)
ax1.barh(mejores10Carpetas,mejores10,color='green')
for i, v in enumerate(mejores10.astype(np.float16)):
    plt.text(v,i,str(v),color='blue',fontsize='xx-small')
# plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
plt.show(fig1)

# fig2 = plt.figure()
# ax2 = fig2.add_axes([0,0,2,2])
fig2,ax2=plt.subplots(dpi=300)
ax2.barh(peores10Carpetas,peores10,color='gray')
for i, v in enumerate(peores10.astype(np.float16)):
    plt.text(v,i,str(v),color='blue',fontsize='xx-small')
# plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
plt.show(fig2)

#%% mejoresxBloque

# A,B o C
bloqueReten = 'A'
bloq=[]
# Bloques = np.zeros((len(listaCarpetas),2))
if bloqueReten == 'A':
    ret1 = 0
    ret2 = 1
for i,carpeta in enumerate(listaCarpetas):
    # Bloques[i,0] = carpeta.split('_')[ret1]
    # Bloques[i,1] = carpeta.split('_')[ret2]
    bloq.append(str(carpeta.split('_')[ret1])+'_'+str(carpeta.split('_')[ret2]))
indices = [i for i,s in enumerate(listaCarpetas) if '1_200_' in s]

listCar = np.asarray(listaCarpetas)
listCar =  listCar[indices]
path_= 'prueba_bucle'
print("Total de entrenamientos a revisar: ",len(listCar))
_predicciones = '_predicciones.npy'
_etiquetas    = '_etiquetasY.npy'
medias = np.zeros(len(listCar))
for n,carpeta in enumerate(listCar):
    path_1 = os.path.join(path_,carpeta)
    print(path_1)
    if 'resultados.txt' in os.listdir(path_1):
        valores = np.genfromtxt(os.path.join(path_,carpeta,'resultados.txt'),dtype=float)
        medias[n] = (sum(valores[2,:])/7+sum(valores[3,:])/7)/2
    # print(valores[0,1])
    # break
mediasOrdenadas = np.sort(medias)
indicesMediasOrdenadas = np.argsort(medias)

mejores10 =mediasOrdenadas[-20:]
mejores10Carpetas = np.array(listCar)[np.array(indicesMediasOrdenadas[-20:])]

peores10 =mediasOrdenadas[:20]
peores10Carpetas = np.array(listCar)[np.array(indicesMediasOrdenadas[:20])]

import matplotlib.pyplot as plt
fig1,ax1=plt.subplots(dpi=300)
ax1.barh(mejores10Carpetas,mejores10,color='green')
for i, v in enumerate(mejores10.astype(np.float16)):
    plt.text(v,i,str(v),color='blue',fontsize='xx-small')
# plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
plt.show(fig1)

# fig2 = plt.figure()
# ax2 = fig2.add_axes([0,0,2,2])
fig2,ax2=plt.subplots(dpi=300)
ax2.barh(peores10Carpetas,peores10,color='gray')
for i, v in enumerate(peores10.astype(np.float16)):
    plt.text(v,i,str(v),color='blue',fontsize='xx-small')
# plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
plt.show(fig2)