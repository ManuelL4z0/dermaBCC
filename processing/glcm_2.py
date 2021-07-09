# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:44:16 2021

@author: ManuelL
"""

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import os
import colorspacious as csp
from skimage.feature import greycomatrix,greycoprops
from funciones_procesado import extrae_parametros_matriz_color,calcula_varias_matrices_coocurrencia4
from colour.appearance import CAM16_VIEWING_CONDITIONS, InductionFactors_CAM16
from colour import CAM16UCS_to_JMh_CAM16
import colour as clr

path1 = 'recortes_LAB_20_aumentados_3'
# path1 = 'recortes_aumentados_2'

path_out = 'VOF_GLCM_LAB_solape_aumentados_3_txt'
tamaños_reglas = [5,7,9,11,13]
# angulos_reglas = [0,45,90,135,180,225,-90,-45]
angulos_reglas = [0,45,90,-45]


listaImagenes = os.listdir(path1)


#%% PARA LA VERSIÓN CON DIFERENCIAS DE ÍNDICES DE LA L (GLCMS HABITUALES)
propi = []
distancias = []
propiedades = ['contrast','dissimilarity','homogeneity','energy','correlation']
for fichero in listaImagenes:
    img = np.load(os.path.join(path1,fichero),allow_pickle=True)
    img = np.asarray(Image.fromarray(img).resize((256,256)))
    imgLAB = csp.cspace_convert(img,"sRGB255","CIELab")
    imgL = imgLAB[:,:,0].astype(np.uint8)
    glcms = greycomatrix(imgL,distances=[1,2,3,4,5],angles=[0,np.pi/4,np.pi/2,-np.pi/4,-np.pi/2],
                          normed=True)
    vof = []
    # glcms = glcms
    glcms2 = np.zeros((256,256,5,3))
    for j in range(5):  #distancias!
        glcms2[:,:,j,0]=glcms[:,:,j,0]
        glcms2[:,:,j,1]=glcms[:,:,j,1]+glcms[:,:,j,3]
        glcms2[:,:,j,2]=glcms[:,:,j,2]+glcms[:,:,j,4]
    for propiedad in propiedades:
        # prop = greycoprops(glcms,propiedad)
        # for j in range(len(prop)):
            # distancias = [prop[j,0],prop[j,1]+prop[j,3],prop[j,2]+prop[j,4]]
            # propi.append(distancias)
        vof.append(greycoprops(glcms2,propiedad))
        # vof.append(propi)
        # propi=[]
    vof = np.asarray(vof)
    vof = vof.reshape((1,5*5*3))
    
    np.savetxt(os.path.join(path_out,fichero[:-4]+'.txt'),vof)
    #orden: contraste:distancia1:angulos:0,pi/4+-pi/4,pi/2+-pi/2,distancia2:repite angulos...
    
#%% PARA LA VERSIÓN CON DIFERENCIAS DE COLORES Y NO DE ÍNDICES
# SE DEBE LEER UN FICHERO CON LOS COLORES
# centroides = np.genfromtxt('centrosColor_20_LAB_RA.txt',dtype=np.float64)
# # centroides = csp.cspace_convert(centroides,"CIELab","sRGB255")
# surr = CAM16_VIEWING_CONDITIONS['Dim']
# coeficientes = InductionFactors_CAM16(0.9,0.59,0.9)
# centroides  = clr.convert(centroides,'CAM16UCS','sRGB',
#           CAM16UCS_to_JMh_CAM16={'surround':surr,' coefficients':coeficientes})

# glcms =[]
# for fichero in listaImagenes:
#     img = np.load(os.path.join(path1,fichero),allow_pickle=True)
    
#     imgLAB = img
#     imgCodi = imgLAB
#     cmedio = [np.mean(imgCodi[:,:,0]),np.mean(imgCodi[:,:,1]),np.mean(imgCodi[:,:,2])]
#     imgL = imgLAB[:,:,0].astype(np.uint8)
    
#     # glcms = greycomatrix(imgL,distances=[1,2,3,4,5],angles=[0,np.pi/4,np.pi/2,-np.pi/4,-np.pi/2],
#     #                      normed=True)
#     print(fichero,img.shape)
#     glcms = calcula_varias_matrices_coocurrencia4(imgL,20,tamaños_reglas,angulos_reglas)
#     parametros = []
#     for glcm in glcms:
#         parametros.append(extrae_parametros_matriz_color(glcm,cmedio,centroides))

#     vof = np.reshape(np.asarray(parametros),(1,140))
#     # break
#     # vof = []
#     # for propiedad in propiedades:
#     #     vof.append(greycoprops(glcms,propiedad))
#     # vof = np.asarray(vof)
#     # vof = vof.reshape((1,6*5*5))
#     np.save(os.path.join(path_out,fichero[:-4]),vof)
    

#%% SUMA DE PROPIEDADES ESPECIFICAS -PRUEBAS-

# propi = []
# distancias = []
# propiedades = ['contrast','dissimilarity','homogeneity','energy','correlation']
# for fichero in listaImagenes:
#     img = np.load(os.path.join(path1,fichero),allow_pickle=True)
#     img = np.asarray(Image.fromarray(img).resize((256,256)))
#     imgLAB = csp.cspace_convert(img,"sRGB255","CIELab")
#     imgL = imgLAB[:,:,0].astype(np.uint8)
#     glcms = greycomatrix(imgL,distances=[1,2,3,4,5],angles=[0,np.pi/4,np.pi/2,-np.pi/4,-np.pi/2],
#                           normed=True)
#     vof = []
#     for propiedad in propiedades:
#         prop = greycoprops(glcms,propiedad)
#         for j in range(len(prop)):
#             distancias = [prop[j,0],prop[j,1]+prop[j,3],prop[j,2]+prop[j,4]]
#             propi.append(distancias)
#         # vof.append(greycoprops(glcms,propiedad))
#         vof.append(propi)
#         propi=[]
#     vof = np.asarray(vof)
#     vof = vof.reshape((1,5*5*3))
    
#     np.save(os.path.join(path_out,fichero[:-4]),vof)
    
