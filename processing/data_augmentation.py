# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 15:15:25 2021

@author: ManuelL
"""

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
from numpy import expand_dims
from PIL import Image
import numpy as np
# import numpy.core.open as openn

path_imagenes ='recortes_LAB_20_3'
path_guardar = 'recortes_LAB_20_aumentados_3'
print("Cantidad de imagenes: ",len(os.listdir(path_imagenes)))

cantidad_imagenes = 4
ordenImagenesYpatrones = np.genfromtxt('ordenImagenesyPatrones_datos_entrenamiento_5.txt',
                                       dtype=str,delimiter=",")
# data_generator= ImageDataGenerator(rotation_range=10,horizontal_flip=True,
#                                    vertical_flip=True)
listaFicheros = os.listdir(path_imagenes)
val=0
for i,nombreFichero in enumerate(listaFicheros):
  # print(imagen)
  imagen = str(i)+'.npy'
  
  # img = np.asarray(Image.open(os.path.join(path_imagenes,nombreFichero)))
  img=np.load(os.path.join(path_imagenes,imagen),allow_pickle=True).astype(np.uint8)

  img = np.asarray(Image.fromarray(img).resize((256,256),Image.NEAREST))
  if(len(np.unique(img))>60):
      print(nombreFichero, "Es un caso especial")  
  img2=img
  for k in range(4):
      if k ==0:
          img2=np.rot90(img)
      else:    
          img2=np.rot90(img2)
      nombreImagen = str(i)+'_'+str(k)
       # print(imagen,nombreImagen)
      np.save(os.path.join(path_guardar,nombreImagen),img2)
      # Image.fromarray(img2.astype(np.uint8)).save(os.path.join(path_guardar,nombreImagen))
  # img = expand_dims(np.asarray(Image.open(os.path.join(path_imagenes,imagen))),0)
  # data_generator.fit(img)
  # print(imagen)
#  creciente = 0





#   for x,val in zip(data_generator.flow(img,seed=val),range(cantidad_imagenes)):
# #    creciente +=1
#     #print(val)
#     print(i,end='')
#     #print(x[0].shape)
#     #print(x[0])
#     nombreImagen = str(i)+'_'+str(int(float(val)))
#     # Image.fromarray(x[0].astype(np.uint8)).save(os.path.join(path_guardar,nombreImagen))
#     np.save(os.path.join(path_guardar,nombreImagen),x[0])
#   print("\n")
