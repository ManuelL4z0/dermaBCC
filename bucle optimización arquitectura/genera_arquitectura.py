# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:24:38 2021

@author: ManuelL
"""
    
def genera_arquitectura(tBA,nBA,tBB,nBB,tBC,nBC,len_test_set_x,decreciente):
    # len_test_set_x = len(test_set_mc_x[0])
    input_shape = (256,256,3)
    from tensorflow.python.keras.applications import vgg16
    from tensorflow.python.keras.models import Sequential,Model
    from tensorflow.python.keras.layers import Dense ,concatenate, Conv2D,Flatten , InputLayer, ZeroPadding2D, Activation, MaxPooling2D,Dropout, GlobalAveragePooling2D, Input

    vgg1=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=input_shape)      
    vgg2=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=input_shape)      
     
    mlpInput = Input(shape=(len_test_set_x,))
    for a,b in enumerate(range(tBB),start=1):
        if decreciente == True:
            print(nBB,a)
            mlp = Dense(int(int(nBB)/a),activation = 'relu')(mlpInput)
        else:
            mlp = Dense(nBB,activation='relu') (mlpInput)            
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
    
    x = GlobalAveragePooling2D(name='avg_pool1')(x)
    y = GlobalAveragePooling2D(name='avg_pool2')(y)

    q = Dense(nBA,activation = 'relu') (y)
    z = Dense(nBA,activation = 'relu') (x)
    
    for a,b in enumerate(range(tBA),start=1):
        if decreciente == True:
            q = Dense(int(int(nBA)/a),activation = 'relu') (q)
            z = Dense(int(int(nBA)/a),activation = 'relu') (z)

        else:
            q = Dense(nBA,activation = 'relu') (q)
            z = Dense(nBA,activation = 'relu') (z)            
    model1 = Model(inputs=vgg1.input, outputs=z)
    model2 = Model(inputs=vgg2.input, outputs=q)

    combinacion = concatenate([model1.output,model2.output,modelMLP.output])

    combinacion = Dense(nBC,activation = 'relu') (combinacion)
    predictZ = Dense(7, activation='sigmoid',name="salida_CNN")(combinacion)
    
    model = Model(inputs=[vgg1.input,vgg2.input,modelMLP.input], outputs=predictZ)
    return model