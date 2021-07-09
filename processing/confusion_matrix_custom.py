# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:11:52 2020

@author: mann
"""
import numpy as np
def confusion_matriz(test_pred,test_set_y):
    tam = len(test_pred)
    predicciones = np.reshape(test_pred.astype(int),(tam,7))
    ground_truth = np.reshape(test_set_y.astype(int),(tam,7))
    tabla_conteo = np.zeros((7,4)).astype(int)
    for i in range(tam):
        for j in range(7):  
            # print(predicciones[i,j],ground_truth[i,j])
            if(predicciones[i,j]==ground_truth[i,j]==0):
                #verdadero negativo
                # print("llega")
                # print(tabla_conteo[j,0],j)
                tabla_conteo[j,0]=tabla_conteo[j,0]+1
                # print(tabla_conteo[j,0])
            if(predicciones[i,j]==ground_truth[i,j]==1):
                #verdadero positivo
                tabla_conteo[j,1]=tabla_conteo[j,1]+1
            if(predicciones[i,j]==1 and ground_truth[i,j]==0):
                #falso positivo
                tabla_conteo[j,2]=tabla_conteo[j,2]+1
            if(predicciones[i,j]==0 and ground_truth[i,j]==1):
                #falso negativo
                tabla_conteo[j,3]=tabla_conteo[j,3]+1
    # print(tabla_conteo)
    return tabla_conteo

def calculo_valores(tabla_conteo):
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

def predict(test_set_x,model):
    pred=model.predict(test_set_x/255,verbose=1)# num_test_images//batch_size+1)
    test_pred=[]
    for i in range(len(pred)):
        test_pred_It = []
        for k in range(7):
            if pred[i,k]>=0.5:
                lession=1
            else:
                lession=0
            test_pred_It.append(lession)
        test_pred.append(test_pred_It)
    test_pred = np.asarray(test_pred)
    test_pred = np.reshape(test_pred,(len(test_set_x),7,1)).astype(int)
    # test_set_y = np.reshape(test_set_y,(482,7,1)).astype(int)
    # print(test_pred.shape,test_set_y.shape)
    # cm = multilabel_confusion_matrix(test_set_y,test_pred,labels=patrones2)    
    # print(cm)
    return test_pred

def predict_dual(test_set_x,test_set_x_2,model):
    pred=model.predict([test_set_x/255,test_set_x_2/255],verbose=1)# num_test_images//batch_size+1)
    test_pred=[]
    for i in range(len(pred)):
        test_pred_It = []
        for k in range(7):
            if pred[i,k]>=0.26:
                lession=1
            else:
                lession=0
            test_pred_It.append(lession)
        test_pred.append(test_pred_It)
    test_pred = np.asarray(test_pred)
    test_pred = np.reshape(test_pred,(len(test_set_x),7,1)).astype(int)
    # test_set_y = np.reshape(test_set_y,(482,7,1)).astype(int)
    # print(test_pred.shape,test_set_y.shape)
    # cm = multilabel_confusion_matrix(test_set_y,test_pred,labels=patrones2)    
    # print(cm)
    return test_pred

def predict_triple(test_set_x,test_set_x_2,test_set_mc_x,model):
    print(len(test_set_x),len(test_set_x_2),len(test_set_mc_x))
    pred=model.predict([test_set_x/255,test_set_x_2/255, test_set_mc_x],verbose=1)# num_test_images//batch_size+1)
    test_pred=[]
    print(pred.shape)
    for i in range(len(pred)):
        test_pred_It = []
        for k in range(7):
            if pred[i,k]>=0.5:
                lession=1
            else:
                lession=0
            test_pred_It.append(lession)
        test_pred.append(test_pred_It)
    test_pred = np.asarray(test_pred)
    print(test_pred.shape)
    test_pred = np.reshape(test_pred,(len(test_set_x),7,1)).astype(int)
    # test_set_y = np.reshape(test_set_y,(482,7,1)).astype(int)
    # print(test_pred.shape,test_set_y.shape)
    # cm = multilabel_confusion_matrix(test_set_y,test_pred,labels=patrones2)    
    # print(cm)
    return test_pred

def predict_triple2(pred_test,umbral):
    # print(len(test_set_x),len(test_set_x_2),len(test_set_mc_x))
    # pred=model.predict([test_set_x/255,test_set_x_2/255, test_set_mc_x],verbose=1)# num_test_images//batch_size+1)
    pred= pred_test
    test_pred=[]
    # print(pred.shape)
    for i in range(len(pred)):
        test_pred_It = []
        for k in range(7):
            if pred[i,k]>=umbral:
                lession=1
            else:
                lession=0
            test_pred_It.append(lession)
        test_pred.append(test_pred_It)
    test_pred = np.asarray(test_pred)
    # print(test_pred.shape)
    test_pred = np.reshape(test_pred,(len(pred_test),7,1)).astype(int)
    # test_set_y = np.reshape(test_set_y,(482,7,1)).astype(int)
    # print(test_pred.shape,test_set_y.shape)
    # cm = multilabel_confusion_matrix(test_set_y,test_pred,labels=patrones2)    
    # print(cm)
    return test_pred

def predict_triple3(pred_test,umbral):
    # print(len(test_set_x),len(test_set_x_2),len(test_set_mc_x))
    # pred=model.predict([test_set_x/255,test_set_x_2/255, test_set_mc_x],verbose=1)# num_test_images//batch_size+1)
    pred= pred_test
    test_pred=[]
    # print(pred.shape)
    for i in range(len(pred)):
        test_pred_It = []
        for k in range(7):
            if pred[i,k]>=umbral[k]:
                lession=1
            else:
                lession=0
            test_pred_It.append(lession)
        test_pred.append(test_pred_It)
    test_pred = np.asarray(test_pred)
    # print(test_pred.shape)
    test_pred = np.reshape(test_pred,(len(pred_test),7,1)).astype(int)
    # test_set_y = np.reshape(test_set_y,(482,7,1)).astype(int)
    # print(test_pred.shape,test_set_y.shape)
    # cm = multilabel_confusion_matrix(test_set_y,test_pred,labels=patrones2)    
    # print(cm)
    return test_pred
