# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:41:20 2021

@author: ManuelL
"""


import os 
import numpy as np
import PIL.Image as Image
from funciones_procesado import convierte_lab, datos_imagen
import colorspacious as csp
import colour as clr
from colour.appearance import CAM16_VIEWING_CONDITIONS, InductionFactors_CAM16
from colour import JMh_CAM16_to_CAM16UCS
# from colorspacious import CIECAM02

path_in = 'datos_entrenamiento_2'
path_out = 'recortes_CAM16_2'

listaImagenes = os.listdir(path_in)

surr = CAM16_VIEWING_CONDITIONS['Dim']
coeficientes = InductionFactors_CAM16(0.9,0.59,0.9)
for img in listaImagenes:
    imagen = Image.open(os.path.join(path_in,img))
    # print(img)
    imagen2=np.asarray(imagen)
    imagen3= np.asarray(imagen).astype(np.float64)
    # imagenLAB = csp.cspace_convert(imagen2,"sRGB255","CIELab")
    
    imgCAM16 = clr.convert(imagen3/255,'sRGB','CAM16UCS',verbose={'mode': 'Short', 'width': 75},
                            JMh_CAM16_to_CAM16UCS={'surround':surr,' coefficients':coeficientes})
    # imagenRGB = csp.cspace_convert(imagenLAB,"CIELab","sRGB255")
    
    
    ## para visualizacion
    # imagenResultado = imgCAM16
    # imgRGB  = clr.convert(imgCAM16,'CAM16UCS','sRGB')
    # Image.fromarray((imgRGB*255).astype(np.uint8)).show()
    
    
    # Image.fromarray(imagenRGB.astype(np.uint8)).show()
    

    # imagenLAB = convierte_lab(np.asarray(imagen))
    # print(imagenLAB)
    # Image.fromarray(imagenLAB).save(os.path.join(path_out,img))
    
    ##comprobaciones!!!!
    # print(img, " Valores J' pertenecen a: [",np.min(imgCAM16[:,:,0]),",",np.max(imgCAM16[:,:,0]),"]")
    # print(img, " Valores a' pertenecen a: [",np.min(imgCAM16[:,:,1]),",",np.max(imgCAM16[:,:,1]),"]")
    # print(img, " Valores b' pertenecen a: [",np.min(imgCAM16[:,:,2]),",",np.max(imgCAM16[:,:,2]),"]")
    # para convertir a CIECAM02
    # datos_imagen(imagen2)
    # datos_imagen(imgRGB)
    # ccam02_space = {"name": "JCh",
                    # "ciecam02_space":"CIECAM02.sRGB"}
    # JCh = csp.cspace_convert(np.asarray(imagen),"sRGB255","JCh")
    # print(np.min(JCh))
    # print(JCh.shape)
    # # print("hola")
    imagenResultado = imgCAM16
    np.save(os.path.join(path_out,img),np.asarray(imagenResultado))
    
    # Image.fromarray(imgRGB.astype(np.uint8)).save(os.path.join(path_out,img))

# https://colorspacious.readthedocs.io/en/latest/reference.html#conversion-functions
    # Image.fromarray(jp).show()
    # break

