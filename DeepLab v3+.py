import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from keras import layers,models,callbacks,backend
import tensorflow as tf
import DataGenerater_unet
import cv2
from skimage.external import tifffile as tiff
import numpy as np
import math
import matplotlib.pyplot as plt

#Xception
def EntryFlow(layer_input,layer_skip,filters):
    
    layer_skip=layers.Conv2D(filters,kernel_size=1,strides=2,padding="same",activation="relu",kernel_initializer="he_normal")(layer_input)
    # one separable_depthwise
    layer1_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer_input)
    layer1_BN=layers.BatchNormalization()(layer1_depth)
    layer1_separa=layers.SeparableConv2D(filters,kernel_size=1,padding="same",activation="relu",kernel_initializer="he_normal")(layer1_BN)
    
    layer2_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer1_separa)
    layer2_BN=layers.BatchNormalization()(layer2_depth)
    layer2_separa=layers.SeparableConv2D(filters,kernel_size=1,padding="same",activation="relu",kernel_initializer="he_normal")(layer2_BN)
    
    layer3_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer2_separa)
    layer3_BN=layers.BatchNormalization()(layer3_depth)
    layer3_separa=layers.SeparableConv2D(filters,kernel_size=3,strides=2,padding="same",activation="relu",kernel_initializer="he_normal")(layer3_BN)
    
    print(layer3_separa.shape,layer_skip.shape)
    block_out=layers.add([layer_skip,layer3_separa])
    
    return block_out,layer_skip

def MiddleFlow_unit(layer_input,filters):
    layer_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer_input)
    layer_BN=layers.BatchNormalization()(layer_depth)
    layer_separa=layers.SeparableConv2D(filters,kernel_size=1,padding="same",activation="relu",kernel_initializer="he_normal")(layer_BN)
    
    return layer_separa

def MiddleFlow(layer_input,layer_skip,filters):
    for i in range(3):
        layer_input=MiddleFlow_unit(layer_input,256)
    layer_out=layers.add([layer_skip,layer_input])
    
#     print(layer_out.shape)
    return layer_out

def ExitFlow(layer_input):
    
    layer_skip=layers.Conv2D(filters=512,kernel_size=1,strides=2,padding="same",activation="relu",kernel_initializer="he_normal")(layer_input)
    
    layer1_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer_input)
    layer1_BN=layers.BatchNormalization()(layer1_depth)
    layer1_separa=layers.SeparableConv2D(filters=256,kernel_size=1,padding="same",activation="relu",kernel_initializer="he_normal")(layer1_BN)
    
    layer2_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer1_separa)
    layer2_BN=layers.BatchNormalization()(layer2_depth)
    layer2_separa=layers.SeparableConv2D(filters=512,kernel_size=1,padding="same",activation="relu",kernel_initializer="he_normal")(layer2_BN)
    
    layer3_depth=layers.DepthwiseConv2D(kernel_size=3,padding="same",activation="relu",kernel_initializer="he_normal")(layer2_separa)
    layer3_BN=layers.BatchNormalization()(layer3_depth)
    layer3_separa=layers.SeparableConv2D(filters=512,kernel_size=1,strides=2,padding="same",activation="relu",kernel_initializer="he_normal")(layer3_BN)
    
    layer_process=layers.add([layer3_separa,layer_skip])
    
    
    layer1=MiddleFlow_unit(layer_process,1024)
    layer2=MiddleFlow_unit(layer1,1024)
    layer_out=MiddleFlow_unit(layer2,2048)
    
    return layer_out

def DCNN(layer_input,layer_skip,filter_base):
    
    for i in range(2):
        layer_input,layer_skip=EntryFlow(layer_input,layer_skip,filter_base*pow(2,(i+1)))
        if(i==0):
            LLFeature=layer_input
        #print(layer_input.shape)
        
#Middle flow
    for j in range(16):
        layer_input=MiddleFlow(layer_input,layer_input,256)
        
    #layer_input is the result of Middle flow
    
#Exit flow
    layer_out=ExitFlow(layer_input)

        
    #print(layer_out.shape)
    print("1. DCNN is over")
    return layer_out,LLFeature
    
def SPP(SPP_input):
    b0=layers.Conv2D(filters=256,kernel_size=1,padding="same",activation="relu",kernel_initializer="he_normal")(SPP_input)
    b1=layers.Conv2D(filters=256,kernel_size=3,padding="same",dilation_rate=6,kernel_initializer="he_normal",activation="relu")(SPP_input)
    b2=layers.Conv2D(filters=256,kernel_size=3,padding="same",dilation_rate=12,kernel_initializer="he_normal",activation="relu")(SPP_input)
    b3=layers.Conv2D(filters=256,kernel_size=3,padding="same",dilation_rate=18,kernel_initializer="he_normal",activation="relu")(SPP_input)

    b4 = layers.GlobalAveragePooling2D()(SPP_input)

    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = layers.Lambda(lambda x: backend.expand_dims(x, 1))(b4)
    #print(b4.shape)
    b4 = layers.Lambda(lambda x: backend.expand_dims(x, 1))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same',use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation(tf.nn.relu)(b4)
    # # upsample. have to use compat because of the option align_corners
#     size_before = layers.Lambda(lambda x:backend.int_shape(x))(SPP_input)
    size_before = backend.int_shape(SPP_input)
#     size_before=layers.Lambda(lambda x:x.shape)(SPP_input)
    print("size_before is ok?")
    b4 = layers.Lambda(lambda x: tf.image.resize_bilinear(x, size_before[1:3], align_corners=True))(b4)
    #print(b0.shape,b2.shape,b2.shape,b3.shape,b4.shape)
    print("2.SPP is over")
    return b0,b1,b2,b3,b4

def Encoder(input_data):
    
    
    # Xception
    x1=layers.Conv2D(filters=32,kernel_size=3,strides=2,padding="same",activation="relu",kernel_initializer="he_normal")(input_data)
    x1=layers.Conv2D(filters=64,kernel_size=3,padding="same",kernel_initializer="he_normal",activation="relu")(x1)

    #DCNN
    SPP_input,LLFeature=DCNN(x1,x1,64)
    
    #SPP
    b0,b1,b2,b3,b4=SPP(SPP_input)
    
    #concat
    layer=layers.concatenate([b0,b1,b2,b3,b4])
    layer=layers.Conv2D(filters=256,kernel_size=1,padding="same",activation="relu")(layer)
    
    print("3.Encoder is over")
    
    return layer,LLFeature

def Decoder(LLFeature,EncoderFeature):
    layerD1=layers.Conv2D(filters=256,kernel_size=1,padding="same",activation="relu")(LLFeature)
    layerD2=layers.UpSampling2D(size=4)(EncoderFeature)
    layer_concat=layers.concatenate([layerD1,layerD2])
    layer=layers.Conv2D(filters=256,kernel_size=3,padding="same",activation="relu")(layer_concat)
    layer=layers.UpSampling2D(size=4)(layer)
    
    print("4.Decoder is over")
    return layer

def DeepLabV3Plus(input_size,num_classes):
    
    input_data=layers.Input(input_size)
    
    EncoderFeature,LLFeature=Encoder(input_data)
    layer=Decoder(LLFeature,EncoderFeature)
    layer=layers.Conv2D(filters=num_classes,kernel_size=1,padding="same",activation="sigmoid")(layer)
    print(layer.shape)
    
    model=models.Model(input_data,layer)
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    
    return model

