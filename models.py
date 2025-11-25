from keras import layers
import tensorflow as tf
import keras


def channel_attention(input_, ratio = 8):
    batch_, w, h, channel = input_.shape
   
    ## Global average pooling
    x1 = layers.GlobalAveragePooling2D()(input_)
    x1 = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x1)
    x1 = layers.Dense(channel, use_bias=False, kernel_initializer='he_normal')(x1) 
    
    ## Max pooling
    x2 = layers.GlobalMaxPool2D()(input_)
    x2 = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x2)
    x2 = layers.Dense(channel, use_bias=False, kernel_initializer='he_normal',)(x2) 
    
    ### Add both and apply sigmoid
    features = x1 + x2
    features = layers.Activation('sigmoid')(features)
    features = layers.multiply([input_, features])
    
    return features

class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def call(self, input_):
        ## average pooling
        x1 = tf.reduce_mean(input_, axis=-1, keepdims=True)

        ## Max pooling
        x2 = tf.reduce_max(input_, axis=-1, keepdims=True)

        ## concatenation
        features = layers.concatenate([x1, x2])
        features = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(features)
        output = layers.multiply([input_, features])
        return output

    def get_config(self):
        config = super().get_config()
        return config

def spatial_attention(input_):
    return SpatialAttention()(input_)

def CBAM(input_): #Convolutional Block Attention Module (CBAM)
    X = channel_attention(input_)
    X = spatial_attention(X)
    return X

def encoder_block(input_, f):
    CN = layers.Conv2D(f, (7,7), activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(input_)
    CN = layers.Conv2D(f, (3,3), activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    
    ### dilatation Conv
    CD = layers.Conv2D(f*2, (3,3), dilation_rate=3, activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    CD = layers.Conv2D(f*2, (3,3), dilation_rate=3, activation='leaky_relu',padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CD)
    CD = layers.Conv2D(f, (3,3), dilation_rate=3, activation='leaky_relu',padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CD)

    C = layers.Conv2D(f, (3,3), activation='leaky_relu', padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    C = layers.concatenate([C, CN])
    C = layers.BatchNormalization()(C)  
   
    skip_connect = layers.Dropout(0.25)(C)
    forward = layers.MaxPooling2D((2,2))(C)
    return forward, skip_connect

def decoder_block(input_, skip_, f):
    U = layers.Conv2DTranspose(f, (2,2), strides=(2,2), activation='leaky_relu', padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(input_)
    C = layers.BatchNormalization()(U) 
    C = layers.Dropout(0.35)(C)
    C = layers.concatenate([C, skip_])   
    C = layers.Conv2D(f, (3,3), dilation_rate=3, activation='leaky_relu', padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(C)
    C = layers.Conv2D(f, (3,3), dilation_rate=3, activation='leaky_relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l1', )(C)
    forward = layers.BatchNormalization()(C)
    return forward

def build_generator(images_shape, f, encoder_trainable = True):
    inp = layers.Input(images_shape) 
    
    encoder_1 = encoder_block(inp, f)
    encoder_2 = encoder_block(encoder_1[0], f)
    encoder_3 = encoder_block(encoder_2[0], f*2)
    encoder_4 = encoder_block(encoder_3[0], f*4)
    encoder_5 = encoder_block(encoder_4[0], f*8)
    encoder_6 = encoder_block(encoder_5[0], f*16)

    
    neck_in = layers.Conv2D(f*32, (3,3), activation='relu', padding='same', name='Model_Neck_1',kernel_initializer='he_normal', kernel_regularizer='l1')(encoder_6[0])
    neck = layers.Conv2D(f*32, (3,3), activation='relu', padding='same', name='Model_Neck_2',kernel_initializer='he_normal', kernel_regularizer='l1')(neck_in)
    neck = layers.Conv2D(f*32, (3,3), activation='relu', padding='same', name='Model_Neck_3',kernel_initializer='he_normal', kernel_regularizer='l1')(neck)
    neck_out = neck

    decoder_6 = decoder_block(neck_out, encoder_6[1], f*16)
    decoder_5 = decoder_block(decoder_6, encoder_5[1], f*8)
    decoder_4 = decoder_block(decoder_5, encoder_4[1], f*4)
    decoder_3 = decoder_block(decoder_4, encoder_3[1], f*2)
    decoder_2 = decoder_block(decoder_3, encoder_2[1], f)
    decoder_1 = decoder_block(decoder_2, encoder_1[1], f)

    output = keras.layers.Conv2DTranspose(1, (2,2), padding='same',kernel_initializer='he_normal', kernel_regularizer='l1')(decoder_1)
    out_image = keras.layers.Activation('sigmoid')(output)
    model = keras.Model(inputs=inp, outputs = out_image) 
    
    if not encoder_trainable:
        for layer in model.layers:
            if layer.name == 'Model_Neck_1':
                break
            layer.trainable = False
         
    return model
