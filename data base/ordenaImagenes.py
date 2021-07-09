# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:11:59 2021

@author: ManuelL
"""

import numpy as np
import PIL.Image as Image
import os

path_in = 'lote imagenes 16 3'
# path_in2= 'recortes_LAB_cod20_aumentados'

# path_out1 = 'recortes_LAB_aumentados_ordenados'
# path_out2 = 'recortes_LAB_cod20_aumentados_ordenados'

patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']
hits =[0,0,0,0,0,0,0]

cadenaPatronesAnterior = np.genfromtxt('ordenImagenesyPatrones_datos_entrenamiento_2.txt' 
                                       ,dtype=str,delimiter=",")    


listaImagenes = os.listdir(path_in)

offsetNombres = len(cadenaPatronesAnterior)
# listaImagenesC= os.listdir(path_in2)
cadenaPatrones = []
for k,nom1 in enumerate(listaImagenes):
    # img1 = np.asarray(np.load(os.path.join(path_in,nom1)))
    # img1 = 
    # errores = []
    listaPatrones=np.zeros(len(patrones2))
    for i,patron in enumerate(patrones2):
        if patron in nom1:
            # hits[i]=int(hits[i]+1)
            listaPatrones[i]=1
            # listaEncoded[k]=int(listaEncoded[k]+np.power(2,i))
    # for nom2 in listaImagenesC:
    #     img2 = np.asarray(Image.open(os.path.join(path_in2,nom2)))
    #     err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    #     err /= float(img1.shape[0] * img2.shape[1])
    #     errores.append(float(err))
    # Image.fromarray(img1).save(os.path.join(path_out1,str(k)+'.jpg'))
    # Image.open(os.path.join(path_in2,listaImagenesC[np.argmin(errores)])).save(os.path.join(path_out2,str(k)+'.jpg'))
    
    cadenaPatrones.append(str(k+offsetNombres)+","+str(listaPatrones)+','+str(nom1))
    # print("Iteracion: ",k," Imagen original: ",nom1," Imagen mas parecida: ",listaImagenes[np.argmin(errores)])
#%%
cadenaNuevaPatrones =cadenaPatronesAnterior.tolist()  + cadenaPatrones

np.savetxt('ordenImagenesyPatrones_datos_entrenamiento_3.txt',cadenaNuevaPatrones,fmt='%str')    
