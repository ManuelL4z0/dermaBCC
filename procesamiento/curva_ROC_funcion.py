# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:32:31 2020

@author: ManuelL
"""

def curva_ROC_multi(test_set_y,test_set_x,test_set_x_2,model,path_out):
    test_pred=model.predict([test_set_x/255,test_set_x_2/255],verbose=1)
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']
    #test pred tienen que ser probabilidades!!
    # fp
    for i in range(test_pred.shape[1]):
        fp,tp,umbrales = roc_curve(test_set_y[:,i],test_pred[:,i])
        print(len(fp),len(tp),len(umbrales))
        auc_ = auc(fp,tp)
        auc_rf = auc_
        fpr_rf = fp
        tpr_rf = tp
        # fpr_keras = 
        
        plt.figure(1,dpi=300)
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.plot(fpr_rf, tpr_rf, label=patrones2[i]+' (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    plt.show()
    nombre1 = "_predicciones"
    nombre2 = "_etiquetasY"
    np.save(os.path.join(path_out,nombre1),test_pred)
    np.save(os.path.join(path_out,nombre2),test_set_y)    
    
def curva_ROC_multi3ple(test_set_y,test_set_x,test_set_x_2,test_set_mc_x,model,path_out):
    test_pred=model.predict([test_set_x/255,test_set_x_2/255,test_set_mc_x],verbose=1)
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']
    #test pred tienen que ser probabilidades!!
    # fp
    for i in range(test_pred.shape[1]):
        fp,tp,umbrales = roc_curve(test_set_y[:,i],test_pred[:,i])
        print(len(fp),len(tp),len(umbrales))
        auc_ = auc(fp,tp)
        auc_rf = auc_
        fpr_rf = fp
        tpr_rf = tp
        # fpr_keras = 
        
        plt.figure(1,dpi=300)
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.plot(fpr_rf, tpr_rf, label=patrones2[i]+' (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    plt.savefig( os.path.join(path_out,'curvas_ROC.png'))    
    plt.show()
    
    nombre1 = "_predicciones"
    nombre2 = "_etiquetasY"
    np.save(os.path.join(path_out,nombre1),test_pred)
    np.save(os.path.join(path_out,nombre2),test_set_y)    
    
def curva_ROC_multi_numPatron(test_set_y,test_set_x,test_set_x_2,numPatron,model):
    test_pred=model.predict([test_set_x/255,test_set_x_2/255],verbose=1)
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt
    patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']
    #test pred tienen que ser probabilidades!!
    # fp
    for i in range(test_pred.shape[1]):
        fp,tp,umbrales = roc_curve(test_set_y[:,i],test_pred[:,i])
        print(len(fp),len(tp),len(umbrales))
        auc_ = auc(fp,tp)
        auc_rf = auc_
        fpr_rf = fp
        tpr_rf = tp
        # fpr_keras = 
        
        plt.figure(1,dpi=300)
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.plot(fpr_rf, tpr_rf, label=patrones2[i+numPatron]+' (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    plt.show()
def curva_ROC_multi_numPatron3ple(test_set_y,test_set_x,test_set_x_2,test_set_mc_x,numPatron,model,path_out):
    test_pred=model.predict([test_set_x/255,test_set_x_2/255,test_set_mc_x],verbose=1)
    print(len(test_pred))
    from sklearn.metrics import roc_curve,auc
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    patrones2= ['PigmentNetwork','Ulceration','Large_B_G_OvoidNests','Multi_B_G_Globules','MapleLeaflike','SpokeWheel','ArborizingTelangiectasia']
    #test pred tienen que ser probabilidades!!
    # fp
    for i in range(test_pred.shape[1]):
        fp,tp,umbrales = roc_curve(test_set_y,test_pred)
        print(len(fp),len(tp),len(umbrales))
        auc_ = auc(fp,tp)
        auc_rf = auc_
        fpr_rf = fp
        tpr_rf = tp
        # fpr_keras = 
        
        plt.figure(1,dpi=300)
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.plot(fpr_rf, tpr_rf, label=patrones2[i+numPatron]+' (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    plt.show()
    nombre1 = patrones2[numPatron]+"_predicciones"
    nombre2 = patrones2[numPatron]+"_etiquetasY"
    np.save(os.path.join(path_out,nombre1),test_pred)
    np.save(os.path.join(path_out,nombre2),test_set_y)
    
    # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # plt.show()