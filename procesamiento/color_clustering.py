# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:32:05 2021

@author: ManuelL
"""

def knn(I,centroides,numCentroides):
    altoImagen,anchoImagen,planos = np.shape(I)
    imagenCodificada = np.zeros(np.shape(I))
    imagenCodificada2 = np.zeros(np.shape(I[:,:,0]))
    # centrosCustom = np.uint8(centroides)
    # centrosCustom = np.uint8(centroides)
    centrosCustom = centroides
    vectorDistancias = np.zeros(numCentroides)
#     print(centrosCustom)
    for i in range(altoImagen):
        for j in range(anchoImagen):
            vectorDistancias = np.zeros(numCentroides)
            puntoImagen=[I[i,j,0],I[i,j,1],I[i,j,2]]
            for k in range(numCentroides):
                vectorDistancias[k]=np.sqrt( np.power((puntoImagen[0]-centrosCustom[k][0]),2)+np.power((puntoImagen[1]-centrosCustom[k][1]),2)
                        +np.power((puntoImagen[2]-centrosCustom[k][2]),2) )
            indiceCentroide = np.argmin(vectorDistancias)
            imagenCodificada[i,j,0] = centrosCustom[indiceCentroide,0]
            imagenCodificada[i,j,1] = centrosCustom[indiceCentroide,1]
            imagenCodificada[i,j,2] = centrosCustom[indiceCentroide,2]
            imagenCodificada2[i,j]   = indiceCentroide
    np.savetxt("imagenCodificadaEtiquetasKNN.txt",imagenCodificada2,fmt='%f')
#     return np.uint8(imagenCodificada)
    return imagenCodificada




#%%
import numpy as np
import os
import PIL.Image as Image
from sklearn.cluster import KMeans

patrones = ['mapleleaf','multiglobules','ovoidnests','pigmentnetwork',
            'spokewheel','telangiectasia','ulceration']
patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests',
            'Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']



path_in = 'recortes_CAM16_2'
path_in2= os.listdir(path_in)
numCentroides = 20    


# listaFicheros = os.listdir(path_in2)

DATOS_IMAGENES = False
# print(np.shape(I))
for i,patron in enumerate(patrones2):
    listaImagenes = []
    path_imagenes = path_in
    print("Path entrada imagenes: ",str(path_imagenes))
    listaFicheros = os.listdir(path_imagenes)
    print(patron)
    print("Cantidad de imagenes: ",len(listaFicheros))
    for k,fichero in enumerate(listaFicheros):
        print(k,"/",len(listaFicheros))
        if patron in fichero:
            if DATOS_IMAGENES == True:
                I = np.asarray(Image.open(os.path.join(path_imagenes,fichero)),np.float32)
            else:
                I = np.load(os.path.join(path_imagenes,fichero),allow_pickle=True)
            listaImagenes.append(np.reshape(I,(I.shape[0]*I.shape[1],3)))
    #print("llega")
    listaImagenes2 = np.reshape(np.asarray(listaImagenes),(len(listaImagenes)*np.shape(I)[0]*np.shape(I)[1],3))
#     print(listaImagenes2[15])
    print("lista completa de ", patron, " tamaño: ",len(listaImagenes))
    kmeans = KMeans(n_clusters=numCentroides,random_state=0,verbose=2,n_init=6, \
                    max_iter = 80).fit(listaImagenes2)
#     print(kmeans.cluster_centers_)
    kmeans_resu_cluster = kmeans.cluster_centers_
    np.savetxt('centroskmeans_'+str(patron)+'_'+str(numCentroides)+'_recortes_CAM16_1.txt',kmeans_resu_cluster)
    
    
#%%
import os    
import numpy as np

path_in = 'centros de color/centros 20 CAM16'
path_out = 'centros de color'
listaFicheros = os.listdir(path_in)
print(listaFicheros)
numCentroides = 20
listaCentroides = []
for fichero in listaFicheros:
    kmeans_resu_cluster = np.genfromtxt(os.path.join(path_in,fichero))
    centroides = np.reshape(kmeans_resu_cluster,(numCentroides,3))
    listaCentroides.append(centroides)
listaCentroides = np.asarray(listaCentroides)
listaCentroides = np.reshape(listaCentroides,(numCentroides*7,3))
k = 0
listaDistancias = []
print("Total de centros: ",len(listaCentroides))
for i in range(len(listaCentroides)):
    centroActual = listaCentroides[i]
    distanciasCentroide = []
    for j in range(i,(len(listaCentroides))):
        distancia=np.sqrt( np.power((centroActual[0]-listaCentroides[j,0]),2)+np.power((centroActual[1]-listaCentroides[j,1]),2)
                        +np.power((centroActual[2]-listaCentroides[j,2]),2) )
        distanciasCentroide.append(np.float64(distancia))
    listaDistancias.append(distanciasCentroide)
#     print(listaDistancias)
print(listaDistancias[0])
listaDistancias = np.asarray(listaDistancias)
np.savetxt(os.path.join(path_out,'listaDistanciasKmeans20CAM16.txt'),listaDistancias,fmt='%s')


#%%

import numpy as np

f = open('centros de color/listaDistanciasKmeans20CAM16.txt')
filas = f.readlines()
# print(filas[5])
print(len(filas))
coloresSimilares = np.zeros((len(filas),len(filas)))
for i,fila in enumerate(filas):
    fila = fila[1:-2].split(',')
    for j,distancia in enumerate(fila[1:]):
#         print(distancia)
        if(float(distancia) < 0.069): #CAM16 0.0605 LAB 10
            print(i,j,distancia)
            coloresSimilares[j,i] = i+1
print("Total de colores: ",len(coloresSimilares))
np.savetxt('matrizColoresSimilaresCAM1620.txt',coloresSimilares,fmt='%f')

#%%

import numpy as np
import operator


coloresSimilares = np.genfromtxt('matrizColoresSimilaresCAM1620.txt',dtype=np.float32)
coloresFinales   = np.ones(len(coloresSimilares[:,0]))
print(coloresSimilares[:,0])
for i in range(len(coloresFinales)):
    print(np.nonzero(coloresSimilares[:,i]))
    indicesTemp = np.nonzero(coloresSimilares[:,i])
    indicesNoSonCero = np.array(indicesTemp) + i
#     indicesNoSonCero = i+np.nonzero(coloresSimilares[:,i])
    print(indicesNoSonCero)
    coloresFinales[indicesNoSonCero] = 0
print(coloresFinales)
print(np.sum(coloresFinales))
np.savetxt('coloresResultadoCAM1620.txt',coloresFinales,fmt='%f',delimiter=',')

#%%

import os
import numpy as np
path_in = 'centros de color/centros 20 CAM16'
listaFicheros = os.listdir(path_in)
print(listaFicheros)
numCentroides = 20
listaCentroides = []
for fichero in listaFicheros:
    kmeans_resu_cluster = np.genfromtxt(os.path.join(path_in,fichero),delimiter = " ")
    centroides = np.reshape(kmeans_resu_cluster,(numCentroides,3))
    listaCentroides.append(centroides)
listaCentroides = np.asarray(listaCentroides)
listaCentroides = np.reshape(listaCentroides,(numCentroides*7,3))
k = 0
listaDistancias = []
print("Total de centros: ",len(listaCentroides))
# print(listaCentroides)
indiceColores = np.genfromtxt('coloresResultadoCAM1620.txt',dtype=np.float32,delimiter=',')
print(sum(indiceColores))
centrosColorFinal = np.zeros((sum(indiceColores.astype(np.int)),3))
for i in range(sum(indiceColores.astype(np.int))):
    centrosColorFinal[i] = listaCentroides[i]

print(centrosColorFinal)
print(len(centrosColorFinal))


#%%

import matplotlib.pyplot as plt
coloresFinales = np.reshape(centrosColorFinal,(20,1,3))
# plt.figure(dpi=300)

# plt.imshow(coloresFinales.astype())
# plt.savefig('coloresFinales_40_LAB.png')
# plt.show()
np.savetxt('centrosColor_20_CAM16_RA.txt',np.reshape(coloresFinales,(20,3)),fmt='%f')

#%%

import PIL.Image as Image
import numpy as np
import os
import colour as clr
import colorspacious as csp
from colour.appearance import CAM16_VIEWING_CONDITIONS, InductionFactors_CAM16
from colour import CAM16UCS_to_JMh_CAM16
path_in  = 'recortes_CAM16_2'
path_out = 'recortes_CAM16_20_2'
centroides = np.genfromtxt('centrosColor_20_CAM16_RA.txt',dtype=np.float64)
centros    = np.reshape(centroides,(20,3))
numCentros = 20
listaImagenes = os.listdir(path_in)
total = len(listaImagenes)
surr = CAM16_VIEWING_CONDITIONS['Dim']
coeficientes = InductionFactors_CAM16(0.9,0.59,0.9)
for i,fichero in enumerate(listaImagenes):
    print(i,"/",total)
    # I = np.asarray(Image.open(os.path.join(path_in,fichero)))
    I = np.load(os.path.join(path_in,fichero),allow_pickle=True)
    imagenCodificada = knn(I,centros,numCentros)
    # imagenRGB = csp.cspace_convert(imagenCodificada,"CIELab","sRGB255")
    # im = Image.fromarray(imagenCodificada.astype(np.uint8)).show()
    # break
    imgRGB  = clr.convert(imagenCodificada,'CAM16UCS','sRGB',
                           CAM16UCS_to_JMh_CAM16={'surround':surr,' coefficients':coeficientes})
#    plt.figure()
    # img = Image.fromarray(imgRGB.astype(np.uint8)).show()
    # break
#    plt.imshow(np.uint8(imagenCodificada))
#    plt.show()
    nombreImagen = str(i)+'.jpg'
    # np.save(os.path.join(path_out,nombreImagen),imagenCodificada)
    imagenResultado = imgRGB
    Image.fromarray((imagenResultado*255).astype(np.uint8)).save(os.path.join(path_out,nombreImagen)) 
    
    
    
#%%
    
import os
import numpy as np
import matplotlib.pyplot as plt
path_in = 'centros de color/centros 18 colores'
listaFicheros = os.listdir(path_in)
numCentroides = 18
listaCentroides = []
plt.figure()
for fichero in listaFicheros:
    kmeans_resu_cluster = np.genfromtxt(os.path.join(path_in,fichero))
    centroides = np.reshape(kmeans_resu_cluster,(numCentroides,1,3))
    plt.imshow(centroides.astype(np.uint8))
    plt.savefig(fichero[0:-11]+'.jpg')
    plt.show()
    
