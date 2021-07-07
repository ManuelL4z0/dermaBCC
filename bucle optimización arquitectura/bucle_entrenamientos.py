# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:36:43 2021

@author: ManuelL
"""

import numpy as np
import os
import PIL.Image as Image
import tensorflow as tf 
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from tensorflow.python.keras.models import Sequential,Model
# import tensorflow.python.keras as K
import tensorflow.python.keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense ,concatenate, Conv2D,Flatten , InputLayer, ZeroPadding2D, Activation, MaxPooling2D,Dropout, GlobalAveragePooling2D, Input
from tensorflow.python.keras.optimizers import Adam,SGD
from tensorflow.compat.v1 import disable_eager_execution
from sklearn.utils import class_weight
import gc
# %% 
config = tf.compat.v1.ConfigProto(gpu_options = 
                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
#%%
path_imagenes2  = 'recortes_aumentados_3'

path_imagenes = 'recortes_RGB_20_3_aumentados'
path_vof = 'VOF_GLCM_LAB_solape_aumentados_3'
ficheroOrdenPatrones = 'ordenImagenesyPatrones_datos_entrenamiento_5.txt'

listaFicheros = os.listdir(path_imagenes)
listaFicherosOriginales = os.listdir(path_imagenes2)

np.random.shuffle(listaFicheros)
lista2 = np.asarray(listaFicheros)
listaEncoded = np.zeros(len(listaFicheros))
ordenPatrones = np.genfromtxt(ficheroOrdenPatrones, dtype = str, delimiter=",")

patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']

totalImagenes = len(listaFicheros)
listaPatrones=np.zeros((totalImagenes,len(patrones2)))

a = []
aOrig=[]
bOrig=[]
cOrig=[]
b = []
c = []
train_set_mc_x = []
test_set_mc_x = []
val_set_mc_x = []
test_set_y  = []
train_set_y = []
val_set_y   = []
nuhsasTest = []
nuhsasVal   = []
print("*Lista de ficheros de longitud: ",len(listaFicheros))

#%%filtrar por patrones

num_train=round(0.7*len(listaFicheros))
num_val=round(0.2*len(listaFicheros))
num_test=round(0.02*len(listaFicheros))

#%% CALCULA PESOS
patronesW = np.zeros(7)
for i in range(len(ordenPatrones)):
      patrones = ordenPatrones[i,1]
      # print(patrones, "  indice: ",indice)
      nuhsa = ordenPatrones[i,2][0:14]
      patronesW[0] += int(patrones[2])
      patronesW[1] += int(patrones[6])
      patronesW[2] += int(patrones[10])
      patronesW[3] += int(patrones[14])
      patronesW[4] += int(patrones[18])
      patronesW[5] += int(patrones[22])
      patronesW[6] += int(patrones[26])   
pesos = 1-(patronesW/len(ordenPatrones))
print(pesos)
#%% CARGA IMAGENES
# listaFicheros = lista3
np.random.shuffle(listaFicheros)
for i,imagen in enumerate(listaFicheros):

  indiceOriginales = os.listdir(path_imagenes2).index(imagen[:-3]+'npy')
  # indiceVOF        = os.listdir(path_vof).index(imagen[:-3]+'txt')
  # vectorCaracteristicas = np.genfromtxt(os.path.join(path_vof,imagen[:-3]+'npy'),
  #                                        dtype=float,delimiter=" ")
  vectorCaracteristicas = np.load(os.path.join(path_vof,imagen[:-3]+'npy'))
  #para leer un vof de una linea:
  # with open(os.path.join(path_vof,imagen[:-3]+'txt')) as f:
  #   vectorCaracteristicas = f.readline()    
  
      
  # print("En esta iteracion toca: ",imagen, indiceOriginales, listaFicherosOriginales[indiceOriginales])
  if indiceOriginales >= 0:
  # print(imagen,i)
      indice = int(float(imagen.split("_")[0]))
      patronesC = np.zeros(len(patrones2))
      patrones = ordenPatrones[indice,1]
      # print(patrones, "  indice: ",indice)
      nuhsa = ordenPatrones[indice,2][0:14]
      patronesC[0] = int(patrones[2])
      patronesC[1] = int(patrones[6])
      patronesC[2] = int(patrones[10])
      patronesC[3] = int(patrones[14])
      patronesC[4] = int(patrones[18])
      patronesC[5] = int(patrones[22])
      patronesC[6] = int(patrones[26])         
      # print(nuhsa)
      # patronesC = int(patrones[14])
      # print(patrones)
      # print(i,"/",len(listaFicheros))
      if i <=num_test:
        im = np.load(os.path.join(path_imagenes,imagen),allow_pickle=True)
  
        c.append(np.asarray(im))
        test_set_y.append(patronesC)
        nuhsasTest.append(nuhsa)

        im2 = np.load(os.path.join(path_imagenes2,listaFicherosOriginales[indiceOriginales]),allow_pickle=True)
        cOrig.append(np.asarray(im2))
        test_set_mc_x.append(vectorCaracteristicas)
        # print("Agrega al test_set la imagen: ",imagen," y ",listaFicherosOriginales[indiceOriginales])
      if num_test<i<num_test+num_val:
        im = np.load(os.path.join(path_imagenes,imagen),allow_pickle=True)
        im2 = np.load(os.path.join(path_imagenes2,listaFicherosOriginales[indiceOriginales]),allow_pickle=True)
        if any(nuhsa in ids for ids in nuhsasTest):
            c.append(np.asarray(im))
            test_set_y.append(patronesC)
            cOrig.append(np.asarray(im2))
            test_set_mc_x.append(vectorCaracteristicas)
            nuhsasTest.append(nuhsa)
        else:
            b.append(np.asarray(im))
            val_set_y.append(patronesC)
            bOrig.append(np.asarray(im2))
            val_set_mc_x.append(vectorCaracteristicas)
            nuhsasVal.append(nuhsa)

      if num_test+num_val<i:
        im = np.load(os.path.join(path_imagenes,imagen),allow_pickle=True)
        im2 = np.load(os.path.join(path_imagenes2,listaFicherosOriginales[indiceOriginales]),allow_pickle=True)
        if any(nuhsa in ids for ids in nuhsasTest):
            c.append(np.asarray(im))
            test_set_y.append(patronesC)
            cOrig.append(np.asarray(im2))
            test_set_mc_x.append(vectorCaracteristicas)
  
        else:            

            a.append(np.asarray(im))
            train_set_y.append(patronesC)        
            train_set_mc_x.append(vectorCaracteristicas)
            aOrig.append(np.asarray(im2))

#%%
#primero los sets con imagenes preprocesadas
test_set_x  = np.reshape(c,[len(c),256,256,3])
test_set_y  = np.reshape(test_set_y,(len(test_set_y),7))
train_set_x = np.asarray(a)
val_set_x   = np.asarray(b)
train_set_y = np.reshape(train_set_y,(len(train_set_y),7))
val_set_y = np.reshape(val_set_y,(len(val_set_y),7))
#seguidamente los sets con imagenes originales
test_set_x_2 = np.asarray(cOrig)
train_set_x_2 = np.asarray(aOrig)
val_set_x_2 = np.asarray(bOrig)

train_set_mc_x = np.asarray(train_set_mc_x).reshape((len(train_set_mc_x),75))
val_set_mc_x = np.asarray(val_set_mc_x).reshape((len(val_set_mc_x),75))
test_set_mc_x = np.asarray(test_set_mc_x).reshape((len(test_set_mc_x),75))
# print(train_set_mc_x.shape)
print("*Carga de imagenes correcta*")

#%%
disable_eager_execution()

img_size=256
input_shape=(img_size,img_size,3)

# adamgrad=Adam(lr=0.00001, beta_1=0.9, beta_2=0.99, epsilon=1e-6,amsgrad=True)

from genera_arquitectura import genera_arquitectura
from tensorflow.python.keras.utils.vis_utils import plot_model

#generar arquitectura: tBA,nBA,tBB,nBB,tBC,nBC,len_test_set_x
nBA = np.linspace(100,300,3).astype(int)
tBA = np.linspace(1,4,4).astype(int)
nBB = np.linspace(10,60,2).astype(int)
tBB = np.linspace(1,5,5).astype(int)
nBC = np.linspace(140,300,2).astype(int)
tBC = np.linspace(1,4,4).astype(int)
# nBA = np.linspace(5,1000,3).astype(int)
# tBA = np.linspace(2,4,3).astype(int)
# nBB = np.linspace(10,60,3).astype(int)
# tBB = np.linspace(2,4,2).astype(int)
# nBC = np.linspace(10,400,2).astype(int)
# tBC = np.linspace(1,2,2).astype(int)

for itBA in tBA:
    for itBB in tBB:
        for itBC in tBC:
            for inBA in nBA:
                for inBB in nBB:
                    for inBC in nBC:
                        decreciente = True
                        vecA = [itBA,inBA,itBB,inBB,itBC,inBC,len(test_set_mc_x[0])]
                        adamgrad= Adam(lr=0.0001)
                        model = genera_arquitectura(vecA[0],vecA[1],vecA[2],vecA[3],vecA[4],vecA[5],vecA[6],decreciente)
                        
                        reduce_lr = ReduceLROnPlateau(monitor='accuracy', 
                                                      factor=0.5, 
                                                      patience=2, 
                                                      verbose=1, 
                                                      mode='max', 
                                                      min_lr=0.0000001)
                        path_out='prueba_bucle'
                        nombreCarpeta = str(vecA[:-1]).replace(', ','_')
                        nombreCarpeta = nombreCarpeta.replace('[','')
                        nombreCarpeta = nombreCarpeta.replace(']','')
                        if decreciente==True:
                            nombreCarpeta = nombreCarpeta+'_cd'
                        try:
                            os.mkdir(os.path.join(path_out,nombreCarpeta))
                        except:
                            indiceNombreCarpeta = len([s for s in os.listdir(path_out) if nombreCarpeta in s])
                            nombreCarpeta = nombreCarpeta+'_ver_'+str(indiceNombreCarpeta)
                            os.mkdir(os.path.join(path_out,nombreCarpeta))
                            print("La carpeta: ",vecA," ya existe, se crea: ",nombreCarpeta)
                            
                        path_out=os.path.join(path_out,nombreCarpeta)
                        earlyStopping = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
                        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
                        model.compile(loss='binary_crossentropy',optimizer=adamgrad,
                                      metrics=['accuracy',tensorflow.keras.metrics.AUC()])
                        plot_model(
                            model,
                            to_file=os.path.join(path_out,"model.png"),
                            show_shapes=True,
                            show_layer_names=True)
                        print("*Comienza el entrenamiento*")
                        
                        #                                 )
                        # for iteri in range(1):
                        modelo_history = model.fit(batch_size=32,x=[(train_set_x)/255,(train_set_x_2)/255,
                                                                    train_set_mc_x],
                                                    y=train_set_y.astype(int),
                                                    epochs=10,verbose=1,
                                                    validation_data=([val_set_x/255,val_set_x_2/255,
                                                                      val_set_mc_x],
                                                                      val_set_y.astype(int)),
                                                    callbacks=[reduce_lr,earlyStopping,mcp_save],
                                                    class_weight = pesos
                                                    )
                        from curva_ROC_funcion import curva_ROC_multi3ple
                        curva_ROC_multi3ple(test_set_y,test_set_x,test_set_x_2,test_set_mc_x,model,
                                                      path_out)
                        # curva_ROC_multi3ple(test_set_y)
                        from confusion_matrix_custom import confusion_matriz,calculo_valores, predict_triple
                        test_pred = predict_triple(test_set_x,test_set_x_2,test_set_mc_x,model)
                        cm = confusion_matriz(test_pred,test_set_y)
                        calculo_valores(cm)
                        del model
                        gc.collect()
                        tf.keras.backend.clear_session()
                        tf.compat.v1.reset_default_graph() 

#%%
