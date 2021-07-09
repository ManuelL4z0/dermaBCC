# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:41:34 2021

@author: ManuelL
"""

import os
import numpy as np
import PIL.Image as Image

path_in = 'reticulos_recortados_201'
path_out= 'datos_entrenamiento_ordenados_4'
patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']

ordenImagenes = np.genfromtxt('ordenImagenesyPatrones_datos_entrenamiento_3.txt',dtype=str,delimiter=',')
offset = int(ordenImagenes.shape[0])
nuevaCadena = []
for i,fichero in enumerate(os.listdir(path_in)):
    img = Image.open(os.path.join(path_in,fichero))
    img.resize((256,256))
    img = np.asarray(img)
    nombre = str(offset+i)
    patrones = "[ 1.  0.  0.  0.  0.  0.  0.]"
    nuevaCadena.append(str(nombre)+','+str(patrones)+','+str(fichero))
    np.save(os.path.join(path_out,nombre),img)
    # break
    
superCadena = ordenImagenes.tolist() + nuevaCadena
np.savetxt('ordenImagenesyPatrones_datos_entrenamiento_5.txt',superCadena,fmt='%str') 
    
    
