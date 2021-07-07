# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:01:18 2021

@author: ManuelL
"""

import numpy as np
import os
import PIL.Image as Image
import tensorflow as tf 
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from tensorflow.python.keras.models import Sequential,Model
import tensorflow.python.keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense ,concatenate, Conv2D,Flatten , InputLayer, ZeroPadding2D, Activation, MaxPooling2D,Dropout, GlobalAveragePooling2D, Input
from tensorflow.python.keras.optimizers import Adam,SGD
from tensorflow.compat.v1 import disable_eager_execution
from sklearn.utils import class_weight
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
# path_imagenes2  = ''
path_imagenes = 'recortes_RGB_20_3_aumentados'
# ficheroDatos = 'ficheros caracteristicas/vectorCaracteristicasPCAMasTAMAS_60.txt'
# ficheroDatos = 'prueba_RGB_aumentados_ordenados.txt'
# ficheroDatos = 'ficheros caracteristicas/VoF_RGB_Recortes_Aumentados_Ordenados60.txt'
path_vof = 'VOF_GLCM_LAB_solape_aumentados_3'
ficheroOrdenPatrones = 'ordenImagenesyPatrones_datos_entrenamiento_5.txt'

listaFicheros = os.listdir(path_imagenes)
listaFicherosOriginales = os.listdir(path_imagenes2)

np.random.shuffle(listaFicheros)
lista2 = np.asarray(listaFicheros)
listaEncoded = np.zeros(len(listaFicheros))

# vectoresCaracteristicas = np.genfromtxt(ficheroDatos,dtype = float , delimiter=" ")
# listaVOF = os.listdir(path_vof)
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
# print("Tamaño del VoF: ",len(vectoresCaracteristicas))


#%%filtrar por patrones
# numPatron = 3
# patronesT = ordenPatrones[:,1]
# patronesC = np.zeros((len(patronesT),len(patrones2)))
# for j,patr in enumerate(patronesT):
#       # patronesC = np.zeros(len(patrones2))
#       # print(nuhsa)
#       # print(j,patr)
#       patronesC[j,0] = int(patr[2])
#       patronesC[j,1] = int(patr[6])
#       patronesC[j,2] = int(patr[10])
#       patronesC[j,3] = int(patr[14])
#       patronesC[j,4] = int(patr[18])
#       patronesC[j,5] = int(patr[22])
#       patronesC[j,6] = int(patr[26])    
#       # listaPatrones.append(patronesC)
# listaPatrones = np.asarray(patronesC)
# listaImagenesConPatron = np.where(listaPatrones[:,numPatron]==1)[0] 
# listaImagenesSinPatron = np.where(listaPatrones[:,numPatron]==0)[0]
# np.random.shuffle(listaImagenesSinPatron)

# lista1 = []
# lista2 = []
# for imgPatron in listaImagenesConPatron:
#     for mm in range(3):
#             imgPatron2 = str(imgPatron)+'_'+str(mm)+'.npy'
#             lista1.append(imgPatron2)
#     # lista1.append(imgAumen)
# for mm in range(3):    
#     for imgSin in listaImagenesSinPatron:
    
#             img32 = str(imgSin)+'_'+str(mm)+'.npy'
#             if(len(lista2)<=len(lista1)*1.25):
#                 lista2.append(img32)
#     # imgSin = str(imgSin)+'_'
#     # imgAumentSin = [igsB for igsB in listaFicheros if imgSin in igsB]
#     # lista2.append(imgAumentSin)
# lista3 = lista2 + lista1

# num_train=round(0.7*len(lista3))
# num_val=round(0.2*len(lista3))
# num_test=round(0.1*len(lista3))
num_train=round(0.7*len(listaFicheros))
num_val=round(0.2*len(listaFicheros))
num_test=round(0.02*len(listaFicheros))

#%%
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
#%%
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

# print("**Recuento**")
# for j in range(7):
    # print("El número de casos de ",patrones2[j]," es de: ",sum(np.asarray(train_set_y[j])))
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
# from tf.keras.optimizers import Adam

# adamgrad=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0,
#               amsgrad=True)

img_size=256
input_shape=(img_size,img_size,3)

# adamgrad=Adam(lr=0.00001, beta_1=0.9, beta_2=0.99, epsilon=1e-6,amsgrad=True)
adamgrad= Adam(lr=0.0001)
opt = SGD(lr=0.001)
# batch_size=12

vgg1=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=input_shape)      
vgg2=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=input_shape)      
  

# modelMLP = Sequential()
# modelMLP.add(Input(shape=(len(test_set_mc_x[0]),))) 
# modelMLP.add(Dense(500,name='mlp_input', activation='relu'))
# modelMLP.add(Dense(500, activation='relu'))
# modelMLP.add(Dense(500, activation='relu'))
# modelMLP.add(Dense(500, activation='relu'))
 
mlpInput = Input(shape=(len(test_set_mc_x[0]),))
mlp = Dense(75,activation='relu') (mlpInput)
# mlp = Dense(100,activation='relu') (mlp)
# mlp = Dense(100,activation='relu') (mlp)
# mlp = Dense(80,activation='relu') (mlp)
# mlp = Dense(80,activation='relu') (mlp)
mlpOutput = Dense(7,activation='sigmoid',name="salida_MLP") (mlp)

modelMLP = Model(inputs=mlpInput,outputs=mlpOutput)

for layer in vgg1.layers:
    layer._name = layer._name + str('_Cc')
for layer in vgg2.layers:
    layer._name = layer._name + str('_Dd')

x = vgg1.output
y = vgg2.output
for nn,layer in enumerate(vgg1.layers):
    layer.trainable=False
    if(nn>=(len(vgg1.layers)-5)):
        layer.trainable=True
for nn,layer in enumerate(vgg2.layers):
    layer.trainable=False
    if(nn>=(len(vgg2.layers)-5)):
        layer.trainable=True

# x = Dropout(0.15)(x)
x = GlobalAveragePooling2D(name='avg_pool1')(x)
# x = Dropout(0.4)(x)
# predict = Dense(7, activation='sigmoid')(x)


# x = Dropout(0.15)(x)
y = GlobalAveragePooling2D(name='avg_pool2')(y)
# y = Dropout(0.4)(y)
# predict = Dense(7, activation='sigmoid')(y)
# model2 = Model(inputs=vgg2.input, outputs=y)

# combinacion = concatenate([model1.output,model2.output,modelMLP.output])
# combinacion = concatenate([model1.output,model2.output])
# z = Dropout(0.15) (combinacion)
# z = Dense(3096,activation = 'relu') (combinacion)
# z = Dropout(0.15)(z)
# z = Dense(3048,activation = 'relu')(z) 
# z = Dropout(0.15)(z)
# z = Dense(2024,activation = 'relu') (z)
# z = Dense(1012,activation = 'relu')(z)
# z = Dense(1500,activation = 'relu') (vgg1.output)
q = Dense(20,activation = 'relu') (y)
# q = Dropout(0.05)(q)
q = Dense(20,activation = 'relu') (q)

z = Dense(20,activation = 'relu') (x)
# z = Dropout(0.05)(z)
z = Dense(20,activation = 'relu') (z)

model1 = Model(inputs=vgg1.input, outputs=z)
model2 = Model(inputs=vgg2.input, outputs=q)
# combinacion = concatenate([model1.output,model2.output])
combinacion = concatenate([model1.output,model2.output,modelMLP.output])
# z = Dropout(0.01)(combinacion)
# combinacion = Dense(3000,activation = 'relu') (combinacion)
# combinacion = Dense(2500,activation = 'relu') (combinacion)
# combinacion = Dense(2000,activation = 'relu') (combinacion)
combinacion = Dense(50,activation = 'relu') (combinacion)
predictZ = Dense(7, activation='sigmoid',name="salida_CNN")(combinacion)

model = Model(inputs=[vgg1.input,vgg2.input,modelMLP.input], outputs=predictZ)
# model = Model(inputs=[vgg1.input,vgg2.input],outputs=[predictZ])
# set_trainable=False
# for layer in vgg.layers:
#     layer.trainable=False
    
reduce_lr = ReduceLROnPlateau(monitor='accuracy', 
                              factor=0.5, 
                              patience=2, 
                              verbose=1, 
                              mode='max', 
                              min_lr=0.0000001)

earlyStopping = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# datagen.fit([train_set_x,train_set_x_2,train_set_mc_x])
model.compile(loss='binary_crossentropy',optimizer=adamgrad,
              metrics=['accuracy',tensorflow.keras.metrics.AUC()])

#model_history=model.fit_generator(train_set,steps_per_epoch=train_steps,
#                                  epochs=10,validation_data=val_set,
#                                  validation_steps=val_steps,verbose=1)   
#%%
# from tpot import TPOTClassifier
# tpot = TPOTClassifier()
# tpot.fit([train_set_x,train_set_x_2,train_set_mc_x],train_set_y)
# print(tpot.score([test_set_x,test_set_x_2,test_set_mc_x],test_set_y))



#spokewheel batchS = 6
#%%

print("*Comienza el entrenamiento*")
# for iteri in range(1):
#     modelo_history = model.fit(batch_size=32,x=[(train_set_x)/255,(train_set_x_2)/255],
#                                 y=train_set_y.astype(int),
#                                 epochs=40,verbose=1,
#                                 validation_data=([val_set_x/255,val_set_x_2/255],
#                                                   val_set_y.astype(int)),
#                                 )
for iteri in range(1):
  modelo_history = model.fit(batch_size=32,x=[(train_set_x)/255,(train_set_x_2)/255,
                                              train_set_mc_x],
                              y=train_set_y.astype(int),
                              epochs=10,verbose=1,
                              validation_data=([val_set_x/255,val_set_x_2/255,
                                                val_set_mc_x],
                                                val_set_y.astype(int)),
                              callbacks=[reduce_lr,earlyStopping,mcp_save],
                              # class_weight = pesos
                              )    
     
# for iteri in range(4):
#     modelo_history = model.fit(batch_size=32,x=[(train_set_x)/255,(train_set_x_2)/255,
#                                                 train_set_mc_x],
#                                 y=[train_set_y.astype(int),train_set_y.astype(int)],
#                                 epochs=40,verbose=1,
#                                 validation_data=([val_set_x/255,val_set_x_2/255,
#                                                   val_set_mc_x],
#                                                   [val_set_y.astype(int),val_set_y.astype(int)]),
#                                 callbacks=[reduce_lr,earlyStopping,mcp_save]
#                                 )    
    
     
# for iteri in range(1):
#     modelo_history = model.fit(batch_size=8,x=(train_set_x_2)/255,
#                                 y=train_set_y.astype(int),
#                                 epochs=40,verbose=1, 
#                                 validation_data=(val_set_x_2/255,val_set_y)
                                
#                                 ) 
#%%
from curva_ROC_funcion import curva_ROC_multi3ple
curva_ROC_multi3ple(test_set_y,test_set_x,test_set_x_2,test_set_mc_x,model,
                              'predicciones/3Entradas_ReticulosCorregidos_Solape/RGB')
# curva_ROC_multi3ple(test_set_y)
from confusion_matrix_custom import confusion_matriz,calculo_valores, predict_triple
test_pred = predict_triple(test_set_x,test_set_x_2,test_set_mc_x,model)
cm = confusion_matriz(test_pred,test_set_y)
calculo_valores(cm)
#%%
#funcion perdidas balanced cross entropy
# import tensorflow.pypthon.keras.backend as K
# def custom_loss():
#     def pbbecc(y_pred,y_true)
# pbbec = -be*yTrue*K.log(yPred)-(1-be)*(1-yTrue)*K.log(1-yPred)



# Precision [0.98113208 0.85377358 0.78459119 0.77672956 0.78930818 0.92138365
#  0.78930818] 
#  VPP [0.94       0.79141104 0.65502183 0.43478261 0.71590909 0.47826087
#  0.73310811] 
#  Especificidad [0.97321429 0.92410714 0.81542056 0.84765625 0.94612069 0.95918367
#  0.78296703] 
#  Sensibilidad [1.         0.68617021 0.72115385 0.48387097 0.36627907 0.45833333
#  0.79779412] 