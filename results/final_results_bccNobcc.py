# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:39:33 2021

@author: Manuel
"""
import numpy as np
import os
from confusion_matrix_custom import confusion_matriz,calculo_valores
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# path_ = 'predicciones/3Entradas/LAB_CCM'
path_ = 'prueba_bucle/1_200_1_35_1_100_Capasdecreciente_ver_2'
_predicciones = '_predicciones.npy'
_etiquetas    = '_etiquetasY.npy'

predicciones = np.load(os.path.join(path_,_predicciones))
etiquetas = np.load(os.path.join(path_,_etiquetas))
tp=0
fn=0
tn=0
fp=0
print("Cantidad de imagenes de Test: ",len(predicciones))
umbrales = np.linspace(0,1,num=20)
vectorUmbrales = [umbrales[17],umbrales[1],umbrales[3],umbrales[4],
                  umbrales[1],umbrales[3],umbrales[10]]

# cm = confusion_matriz(predicciones,etiquetas)
# calculo_valores(cm)
bcc   = False
nobcc = False
bccgt=False
nobccgt=False
diagnostico = False
real = False
resultado = []
# umbrales = np.linspace(0,1,5)
for i in range(len(predicciones)):
    # print(predicciones[i],etiquetas[i])
    if(predicciones[i,0]>vectorUmbrales[0]):
        nobcc=True
    for j,umbral in enumerate(vectorUmbrales[1:],start=1,):
        print(j)
        if(predicciones[i,j]>umbral):
            bcc = True
    # print(nobcc,bcc,predicciones[i],etiquetas[i])
    if(etiquetas[i,0]==1):
        nobccgt = True
    if(any(etiquetas[i,1:]==1)):
        bccgt   = True
    # print("NoBCC dijo: ",nobcc," y era: ",nobccgt)
    # print("BCC dijo: ",bcc," y era: ",bccgt)
    
    if(nobcc == True):
        diagnostico = False
    if(bcc == True):
        diagnostico = True
    if(bccgt == True):
        real = True
    else:
        real = False
        
    # print("Diagnostico: ",diagnostico, " y realmente era: ",real)
    
    if(diagnostico==real==True):
        tp +=1
    if(diagnostico==real==False):
        tn +=1
    if(diagnostico==True and real == False):
        fp +=1
    if(diagnostico==False and real == True):
        fn +=1    
    nobcc=bcc=bccgt=nobccgt=diagnostico=real=False

cm = np.zeros((2,2))
cm[0,0]=tp
cm[0,1]=fn
cm[1,0]=fp
cm[1,1]=tn
# cmap = plt.cm.get_cmap("winter")
nombres =["BCC","NO BCC"]
fig, ax = plt.subplots(dpi=300)

ax.matshow(cm, cmap='gray')

for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

ax.set_xticklabels(['']+nombres)
plt.ylabel("Ground Truth")
plt.xlabel("Predictions")
ax.set_yticklabels(['']+nombres)
plt.show()
# print("Umbral: ",umbral)
print("Exactitud: ",(tp+tn)/(tp+tn+fp+fn))
print("VPP: ",tp/(fp+tp))
print("Especificidad: ",(tn/(tn+fp)))
print("Sensibilidad: ",(tp/(tp+fn)))


# plt.matshow(cm)

# plt.show()
    
    
    
    
    
    
    
    
def calculo_valores2(tabla_conteo):
    tabla_conteo = tabla_conteo
    a,b = tabla_conteo.shape
    
    precision = np.zeros(a) #tp+tn/tp+tn+fp+fn
    vpp       = np.zeros(a) #tp/fp+tp
    especifici= np.zeros(a) #tn/tn+fp
    sensibilid= np.zeros(a) #tp/tp+fn
    total = sum(tabla_conteo[0,:])
    for i in range(a):
        precision[i] = (tabla_conteo[i,0]+tabla_conteo[i,1])/total
        vpp[i]       = tabla_conteo[i,1]/(tabla_conteo[i,1]+tabla_conteo[i,2])
        especifici[i]= tabla_conteo[i,0]/(tabla_conteo[i,0]+tabla_conteo[i,2])
        sensibilid[i]= tabla_conteo[i,1]/(tabla_conteo[i,1]+tabla_conteo[i,3])
    # print("Precision",precision, "\n",
    #       "VPP",vpp, "\n",
    #       "Especificidad",especifici, "\n",
    #       "Sensibilidad",sensibilid, "\n",)
    return precision,vpp,especifici,sensibilid    
    
    
    
    
    
    
    # if predicciones[i,0] >=0.5 and etiquetas[i,0]==1:
    #     nobcc_tp +=1
    #     nobcc= True
    # if predicciones[i,0] >=0.5 and etiquetas[i,0]==0:
    #     nobcc_fp +=1
    # if predicciones[i,0] <0.5 and etiquetas[i,0]==1:
    #     nobcc_fn +=1
    # if predicciones[i,0] >=0.5 and etiquetas[i,0]==0:
    #     nobcc_tn +=1
    # for j in range(1,6):
    #     if predicciones[i,j] >=0.5 and etiquetas[i,j]==1:
    #         bcc_tp +=1
    #         bcc=True
    #     if predicciones[i,j] >=0.5 and etiquetas[i,j]==0:
    #         bcc_fp +=1
    #     if predicciones[i,j] <0.5 and etiquetas[i,j]==1:
    #         bcc_fn +=1
    #     if predicciones[i,j] >=0.5 and etiquetas[i,j]==0:
    #         bcc_tn +=1
#     resultado.append(["BCC: ",bcc, " NO-BCC: ",nobcc," ",etiquetas[i],predicciones[i]])
#     bcc = nobcc = False
# print("NoBcc TP: ",nobcc_tp)
# print("NoBcc FP: ",nobcc_fp)
# print("NoBcc TN: ",nobcc_tn)
# print("NoBcc FN: ",nobcc_fn)
# print("Bcc TP: ",bcc_tp)
# print("Bcc FP: ",bcc_fp)
# print("Bcc TN: ",bcc_tn)
# print("Bcc FN: ",bcc_fn)



