# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:07:25 2020

@author: mann
"""
from skimage import io, color
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import cv2
from scipy.stats import skew
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

path_in  ='datos_entrenamiento'
path_out ='datos_procesados_lab_3'
listaImagenesEntrada = os.listdir(path_in)



caracteristicasDeEntrada=[]

def genera_caracteristicas():
    angulos_reglas = [0,45,90,-45]
    codif = ['1_','2_','3_','4_','5_','6_','7_','8_','9_','10_','11_','12_','13_','14_','15_','16_']
    parametros = ['homogeneidad','glcmMedia','varianza','entropia','correlacion']

    for angulo in angulos_reglas:
        for parametro in parametros:
            caracteristicasDeEntrada.append(parametro+str(angulo))
    colores = ['CanalA','CanalB']
    for color in colores :       
        for cod in codif:
            caracteristicasDeEntrada.append(cod + color)
    return caracteristicasDeEntrada
        

def criba_coeficientes(coeficientes,etiquetas):
    if(len(etiquetas)==len(coeficientes)):
#         np.amax(coeficientes,resul)        
        coefiOrdenados = np.argsort(coeficientes)
        listaCoeficientes2 = coefiOrdenados[0:5]
        listaCoeficientes1 = coefiOrdenados[-5:]
        
        print(listaCoeficientes1,listaCoeficientes2)
        print("Parametros mas relevantes: ")
        for i in range(5):
            print(etiquetas[listaCoeficientes1[i]]," ",coeficientes[listaCoeficientes1[i]])
        print("Parametros menos relevantes: ")
        for i in range(5):
            print(etiquetas[listaCoeficientes2[i]]," ",coeficientes[listaCoeficientes2[i]])    
    else:
    
        print("Error, etiquetas y caracteristicas de tamaños diferentes")
    
# criba_coeficientes(coef,caracteristicasDeEntrada)  

def criba_coeficientes_RFE(X_Train,Y_Train,etiquetas):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=10,step=1)
    selector = selector.fit(X_Train,Y_Train)
    selector.support_
    ranking  = selector.ranking_
    print("Caracteristicas mas relevantes")
    for i in range(len(ranking)):
        if(ranking[i]==1):
            print(etiquetas[i])
            

def calcula_varias_matrices_coocurrencia2(imagen,valores,media,tamaños_reglas,angulos_reglas):
    listaParametros=[]
    listaMatrices  =[]
    #print("Comienza el calculo de las matrices de coocurrencia")
    for tama in tamaños_reglas:
        for angulo in angulos_reglas:
            reglaMatriz = genera_regla(tama,angulo)
            # print(valores,imagen,reglaMatriz)
            matrizCoocurrencia = calcula_matriz_coocurrencia(valores,imagen,reglaMatriz)
            listaMatrices.append(matrizCoocurrencia)
            listaParametros+=extrae_parametros_matriz_gris(matrizCoocurrencia,media,valores)
    return listaParametros,listaMatrices

def calcula_varias_matrices_coocurrencia3(imagen,valores,media,tamaños_reglas,angulos_reglas):
    listaParametros=[]
    lista_matrices = []
    
    for tama in tamaños_reglas:
        for angulo in angulos_reglas:
            lista_matrices.append(genera_regla(tama,angulo))
    GLCMS = calcula_varias_matrices_GLCM(valores,imagen,lista_matrices)
    for glcm in GLCMS:
        # print(tamaños_reglas,angulos_reglas)
        listaParametros += extrae_parametros_matriz(glcm,media,valores)
    print(listaParametros)
    return listaParametros

def calcula_varias_matrices_coocurrencia4(imagen,valores,tamaños_reglas,angulos_reglas):
    listaMatrices  =[]
    #print("Comienza el calculo de las matrices de coocurrencia")
    for tama in tamaños_reglas:
        for angulo in angulos_reglas:
            reglaMatriz = genera_regla(tama,angulo)
            matrizCoocurrencia = calcula_matriz_coocurrencia(valores,imagen,reglaMatriz)
            listaMatrices.append(matrizCoocurrencia)
    return listaMatrices

def extrae_parametros_matriz_gris(matrizCoocurrencia,media,cuantizacion):

    matrizCoocurrenciaProb = probabilidad_matriz_coocurrencia(matrizCoocurrencia,cuantizacion)
    hMatriz,wMatriz = matrizCoocurrenciaProb.shape[:2]
     
    media = np.mean(matrizCoocurrencia)
    homogeneidad = 0
    # contraste    = 0
    glcmMedia    = 0
    varianza     = 0
    entropia     = 0
    for i in range(hMatriz):
        for j in range(wMatriz):
            # print(cuantizacion)
    #         print(np.power((i-j),2))
            homogeneidad += matrizCoocurrenciaProb[i,j]/(1+np.power((i-j),2))
            # contraste    += matrizCoocurrenciaProb[i,j]*np.power((i-j),2)
            glcmMedia    += i*matrizCoocurrenciaProb[i,j]
            varianza     += matrizCoocurrenciaProb[i,j]*np.power((i-media),2)
            if(matrizCoocurrenciaProb[i,j]!=0):
                entropia     += -1*matrizCoocurrenciaProb[i,j]*np.log(matrizCoocurrenciaProb[i,j])

    correlacion = 0
    for i in range(hMatriz):
        for j in range(wMatriz):    
            correlacion += matrizCoocurrenciaProb[i,j]*((i-glcmMedia)*(j-glcmMedia)/
                                                    (np.sqrt(varianza*varianza)))
    listaParametros=[]
#     listaParametros.append("================="+'\n')
#     listaParametros.append("Homogeneidad es: "+str(homogeneidad)+'\n')
    listaParametros.append(homogeneidad)
    listaParametros.append(glcmMedia)
    listaParametros.append(varianza)
    listaParametros.append(entropia)
    listaParametros.append(correlacion)
#     listaParametros.append("Contraste es: "+str(contraste)+'\n')
#     listaParametros.append("GLCM media es: "+str(glcmMedia)+'\n')
#     listaParametros.append("Varianza es: "+str(varianza)+'\n')
#     listaParametros.append("Desviacion tipica es: "+str(np.sqrt(varianza))+'\n')
#     listaParametros.append("Entropia es: "+str(entropia)+'\n')
#     listaParametros.append("Correlacion es: "+str(correlacion)+'\n')
#     listaParametros.append("Media es: "+str(media))
    # print(len(listaParametros))        
    return listaParametros

    
    
def extrae_parametros_matriz_color(matrizCoocurrencia,media,centros):

    matrizCoocurrenciaProb = probabilidad_matriz_coocurrencia(matrizCoocurrencia)
    # matrizCoocurrenciaProb = matrizCoocurrencia
    hMatriz,wMatriz = matrizCoocurrenciaProb.shape[:2]
     
    
    homogeneidad = 0
    # contraste    = 0
    glcmMedia    = [0,0,0]
    varianza     = 0
    entropia     = 0
    for i in range(hMatriz):
        for j in range(wMatriz):
            # print(cuantizacion)
    #         print(np.power((i-j),2))
            homogeneidad += matrizCoocurrenciaProb[i,j]/(1+np.sqrt(np.power((centros[i,0]-centros[j,0]),2)
                                                          +np.power((centros[i,1]-centros[j,1]),2)
                                                          +np.power((centros[i,2]-centros[j,2]),2)))
            # contraste    += matrizCoocurrenciaProb[i,j]*np.power((i-j),2)
            glcmMedia[0]  += matrizCoocurrenciaProb[i,j]*(centros[i,0])
            glcmMedia[1]  += matrizCoocurrenciaProb[i,j]*(centros[i,1])
            glcmMedia[2]  += matrizCoocurrenciaProb[i,j]*(centros[i,2])
            
        
            if(matrizCoocurrenciaProb[i,j]!=0):
                entropia     += -1*matrizCoocurrenciaProb[i,j]*np.log(matrizCoocurrenciaProb[i,j])

    correlacion = 0
    for i in range(hMatriz):
        for j in range(wMatriz):
            varianza     += matrizCoocurrenciaProb[i,j]*(np.power((centros[i,0]-glcmMedia[0]),2)
                                                          +np.power((centros[i,1]-glcmMedia[1]),2)
                                                          +np.power((centros[i,2]-glcmMedia[2]),2))
    for i in range(hMatriz):
        for j in range(wMatriz):    
            correlacion += matrizCoocurrenciaProb[i,j]*((np.sqrt(np.power((centros[i,0]-glcmMedia[0]),2)
                                                          +np.power((centros[i,1]-glcmMedia[1]),2)
                                                          +np.power((centros[i,2]-glcmMedia[2]),2)))
                                                      *(np.sqrt(np.power((centros[j,0]-glcmMedia[0]),2)
                                                          +np.power((centros[j,1]-glcmMedia[1]),2)
                                                          +np.power((centros[j,2]-glcmMedia[2]),2)))/
                                                    (np.sqrt(varianza*varianza)))
            
            
    listaParametros=[]
#     listaParametros.append("================="+'\n')
#     listaParametros.append("Homogeneidad es: "+str(homogeneidad)+'\n')
    listaParametros.append(homogeneidad)
    listaParametros.append(glcmMedia[0])
    listaParametros.append(glcmMedia[1])
    listaParametros.append(glcmMedia[2])
    listaParametros.append(varianza)
    listaParametros.append(entropia)
    listaParametros.append(correlacion)
#     listaParametros.append("Contraste es: "+str(contraste)+'\n')
#     listaParametros.append("GLCM media es: "+str(glcmMedia)+'\n')
#     listaParametros.append("Varianza es: "+str(varianza)+'\n')
#     listaParametros.append("Desviacion tipica es: "+str(np.sqrt(varianza))+'\n')
#     listaParametros.append("Entropia es: "+str(entropia)+'\n')
#     listaParametros.append("Correlacion es: "+str(correlacion)+'\n')
#     listaParametros.append("Media es: "+str(media))
    # print(len(listaParametros))        
    return listaParametros

def probabilidad_matriz_coocurrencia(matriz):
    total=int(sum(sum(matriz)))
#     print(total)
    matrizCoocurrenciaProbabilidades = matriz/total
#     print(matrizCoocurrenciaProbabilidades)
#     print(sum(sum(matrizCoocurrenciaProbabilidades)))
    return matrizCoocurrenciaProbabilidades




def convierte_lab(imagen):
    # img = cv2.imread(imagen) 
    img = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    return img

def kmeans_cluster(fichero,colores):
#     img = np.asarray(Image.open(fichero))
    img=fichero
    Z = img.reshape((-1,1))
#     print(Z.shape)
    Z = np.float32(Z)
    criterios = (cv2.TERM_CRITERIA_EPS, 20000, 0.001)
    K = colores
    ret,label,centroides=cv2.kmeans(Z,K,None,criterios,10,cv2.KMEANS_RANDOM_CENTERS)
    centroides = np.uint8(centroides)
    res = centroides[label.flatten()]
    res2 = res.reshape((img.shape))
#     print("Tamaño del conjunto de centroides ",len(centroides))
#     print(centroides)
#     print(np.unique(centroides))
#     print("Tamaño del conjunto de etiquetas ",len(label))
#     print(label)
#     print("Valores etiquetas ",np.unique(label))
#     return res2
    return label.reshape((img.shape))
    
def kmeans_cluster_centroides(imagen,colores):
#     img = np.asarray(Image.open(fichero))
    #img=fichero
    #Z = np.vstack(lista)
#     print(Z.shape)
    #Z = np.reshape(imagen,(-1,3))
    Z = imagen.reshape(-1,3)
    #print(Z.shape)
    Z = np.float32(Z)
    criterios = (cv2.TERM_CRITERIA_EPS, 200000, 0.001)
    K = colores
    ret,label,centroides=cv2.kmeans(Z,K,None,criterios,100,cv2.KMEANS_RANDOM_CENTERS)
    centroides = np.uint8(centroides)
    res = centroides[label.flatten()]
    res2 = res.reshape((imagen.shape))
#     print("Tamaño del conjunto de centroides ",len(centroides))
#     print(centroides)
#     print(np.unique(centroides))
#     print("Tamaño del conjunto de etiquetas ",len(label))
#     print(label)
#     print("Valores etiquetas ",np.unique(label))
#     return res2
    return centroides    

def calcula_varias_matrices_GLCM(valores,imagen,matrices):
    print("Valores en la imagen:",np.unique(imagen))
    hitsCoocurrencia = np.zeros((valores,valores,len(matrices)))
    hImagen,wImagen = imagen.shape[:2]
    for i in range(hImagen):
        for j in range(wImagen):
            for k,matriz in enumerate(matrices):
                hMatriz,wMatriz = matriz.shape[:2]
                if hImagen-int(hMatriz) >=0 and wImagen-int(wMatriz) >=0 \
                    and hImagen>=int(i+hMatriz) and wImagen>=int(j+wMatriz):
                    # print(i,j,hMatriz,wMatriz,i,j)
                    ventanaImagen = imagen[i:hMatriz+i,j:wMatriz+j]
                    matrizSobreImagen = matriz*ventanaImagen
                    valoresCoocurrencia = matrizSobreImagen[np.nonzero(matriz)]
                    hitsCoocurrencia[int(valoresCoocurrencia[0]),int(valoresCoocurrencia[1]),k]+=int(1)
    np.savetxt('coocurrencia1.txt',hitsCoocurrencia[:,:,0],fmt='%i')
    return hitsCoocurrencia







def calcula_matriz_coocurrencia(valores,imagen,matriz):
    #valores es la cantidad de colores que podría haber diferentes en este plano de la imagen
    hitsCoocurrencia = np.zeros((valores,valores))
    vdiferentes = np.unique(imagen)
    
#     print(hitsCoocurrencia.shape)
    hImagen,wImagen  = imagen.shape[:2]
    hMatriz,wMatriz  = matriz.shape[:2]
    # print(hImagen,wImagen,hMatriz,wMatriz)
    for i in range(hImagen-hMatriz):
        for j in range(wImagen-wMatriz):
            ventanaImagen = imagen[i:hMatriz+i,j:wMatriz+j]
            matrizSobreImagen=matriz*ventanaImagen
            valoresCoocurrencia = matrizSobreImagen[np.nonzero(matriz)]
            #print("valores coocurrencia",valoresCoocurrencia)
#             if(valoresCoocurrencia[0]==0):
#                 print(valoresCoocurrencia)
            # int(valoresCoocurrencia[0])
#             print(valoresCoocurrencia)
            hitsCoocurrencia[int(np.where(vdiferentes==valoresCoocurrencia[0])[0]),
                             int(np.where(vdiferentes==valoresCoocurrencia[1])[0])]+=1
    # print(hitsCoocurrencia)
    return hitsCoocurrencia

def genera_regla(tama,angulo):
    matriz = np.zeros((tama,tama))
#     print(int(np.ceil(tama/2)))
    matriz[int(np.ceil(tama/2))-1,int(np.ceil(tama/2))-1]=1
    if(angulo==0):
        matriz[int(np.ceil(tama/2))-1,tama-1]=1
    if(angulo==180):
        matriz[int(np.ceil(tama/2))-1,0]=1
    if(angulo==-90):
        matriz[tama-1,int(np.ceil(tama/2))-1]=1
    if(angulo==90):
        matriz[0,int(np.ceil(tama/2))-1]=1
    if(angulo==45):
        matriz[0,tama-1]=1
    if(angulo==135):
        matriz[0,0]=1
    if(angulo==225):
        matriz[tama-1,0]=1
    if(angulo==-45 or angulo==315):
        matriz[tama-1,tama-1]=1
    
#     print(matriz)    
    return matriz

def datos_imagen_grey(imagen,valores_cuantizacion):
    print("===================")
    plt.figure
    hist,n = np.histogram(imagen,range(valores_cuantizacion),[0,valores_cuantizacion-1])
    plt.plot(n[1:],hist)
    plt.xlim([0,valores_cuantizacion])
    indice2=0
    for i in n[1:-1]:
#         print(hist[i])
        if(hist[i]!=0):
            b=np.ones(hist[i])*int(i)
#             print(b)
            if(indice2==0):
                indice2=1
                a=b
            else:
                indice2+=1
                a=np.concatenate([a,b])    

    media            = np.mean(a)
    desviacionTipica = np.sqrt(np.var(a))
    skewness         = skew(a)

    print("La media de los valores de intensidad es: ",media)
    print("La desviación típica es: ",desviacionTipica)
    print("La asimetría estadística (skewness) es: ",skewness)
    
    plt.axvline(media,linestyle='dashed',linewidth=2) 
    print("===================")
    plt.show()
    
def datos_imagen(imagen):
    color = ('r','g','b')
    print("===================")
    for i, colo in enumerate(color):
        plt.figure
        hist,n = np.histogram(imagen[:,:,i],range(256),[0,255])
        plt.plot(n[1:],hist,color = colo)
        plt.xlim([0,256])

        indice2=0
        for i in n[1:-1]:
    #         print(hist[i])
            if(hist[i]!=0):
                b=np.ones(hist[i])*int(i)
    #             print(b)
            if(hist[i]!=0):
                if(indice2==0):
                    indice2=1
                    a=b
                else:
                    indice2+=1
                    a=np.concatenate([a,b])    

        media            = np.mean(a)
        desviacionTipica = np.sqrt(np.var(a))
        skewness         = skew(a)

        print("La media del color ",colo," es: ",media)
        print("La desviación típica del color ",colo," es: ",desviacionTipica)
        print("La asimetría estadística (skewness) del color ",colo," es: ",skewness)

        plt.axvline(media,linestyle='dashed',color = colo,linewidth=2) 
        print("===================")
    plt.show()
