from keras import layers
import tensorflow as tf
import keras


def channel_attention_3d(input_, ratio=8):
    """3D Channel Attention Module"""
    batch_, d, w, h, channel = input_.shape

    # Global average pooling
    x1 = layers.GlobalAveragePooling3D()(input_)
    x1 = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x1)
    x1 = layers.Dense(channel, use_bias=False, kernel_initializer='he_normal')(x1)

    # Max pooling
    x2 = layers.GlobalMaxPool3D()(input_)
    x2 = layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x2)
    x2 = layers.Dense(channel, use_bias=False, kernel_initializer='he_normal')(x2)

    # Add both and apply sigmoid
    features = x1 + x2
    features = layers.Activation('sigmoid')(features)
    features = layers.Reshape((1, 1, 1, channel))(features)
    features = layers.multiply([input_, features])

    return features


class SpatialAttention3D(layers.Layer):
    """3D Spatial Attention Module"""
    def __init__(self, **kwargs):
        super(SpatialAttention3D, self).__init__(**kwargs)
        # Create the Conv3D layer once during initialization
        self.conv3d = layers.Conv3D(1, (7, 7, 7), padding='same', activation='sigmoid')

    def call(self, input_):
        # Average pooling across channel dimension
        x1 = tf.reduce_mean(input_, axis=-1, keepdims=True)

        # Max pooling across channel dimension
        x2 = tf.reduce_max(input_, axis=-1, keepdims=True)

        # Concatenation
        features = layers.concatenate([x1, x2])
        features = self.conv3d(features)  # Use the pre-created layer
        output = layers.multiply([input_, features])
        return output

    def get_config(self):
        config = super().get_config()
        return config


def spatial_attention_3d(input_):
    return SpatialAttention3D()(input_)


def CBAM_3d(input_):
    """3D Convolutional Block Attention Module"""
    X = channel_attention_3d(input_)
    X = spatial_attention_3d(X)
    return X


def encoder_block_3d(input_, f):
    """3D Encoder Block"""
    CN = layers.Conv3D(f, (3, 3, 3), activation='leaky_relu', padding='same',
                       kernel_initializer='he_normal', kernel_regularizer='l1')(input_)
    CN = layers.Conv3D(f, (3, 3, 3), activation='leaky_relu', padding='same',
                       kernel_initializer='he_normal', kernel_regularizer='l1')(CN)

    # Dilated Conv
    CD = layers.Conv3D(f*2, (3, 3, 3), dilation_rate=(2, 2, 2), activation='leaky_relu',
                       padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    CD = layers.Conv3D(f*2, (3, 3, 3), dilation_rate=(2, 2, 2), activation='leaky_relu',
                       padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CD)
    CD = layers.Conv3D(f, (3, 3, 3), dilation_rate=(2, 2, 2), activation='leaky_relu',
                       padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(CD)

    C = layers.Conv3D(f, (3, 3, 3), activation='leaky_relu', padding='same',
                      kernel_initializer='he_normal', kernel_regularizer='l1')(CN)
    C = layers.concatenate([C, CN])
    C = layers.BatchNormalization()(C)

    skip_connect = layers.Dropout(0.25)(C)
    forward = layers.MaxPooling3D((2, 2, 2))(C)
    return forward, skip_connect


def decoder_block_3d(input_, skip_, f):
    """3D Decoder Block"""
    U = layers.Conv3DTranspose(f, (2, 2, 2), strides=(2, 2, 2), activation='leaky_relu',
                               padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(input_)
    C = layers.BatchNormalization()(U)
    C = layers.Dropout(0.35)(C)
    C = layers.concatenate([C, skip_])
    C = layers.Conv3D(f, (3, 3, 3), dilation_rate=(2, 2, 2), activation='leaky_relu',
                      padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(C)
    C = layers.Conv3D(f, (3, 3, 3), dilation_rate=(2, 2, 2), activation='leaky_relu',
                      padding='same', kernel_initializer='he_normal', kernel_regularizer='l1')(C)
    forward = layers.BatchNormalization()(C)
    return forward


def build_3d_generator(image_shape, f, encoder_trainable=True, num_modalities=1):
    """
    Build 3D U-Net for multi-modal fat suppression
    Args:
        image_shape: Tuple of (depth, height, width, channels*num_modalities)
        f: Base number of filters
        encoder_trainable: Whether encoder is trainable
        num_modalities: Number of input modalities
    """
    inp = layers.Input(image_shape)

    # Multi-modal fusion at input level
    if num_modalities > 1:
        # Split modalities and apply modality-specific processing
        modalities = []
        channels_per_modality = image_shape[-1] // num_modalities

        for i in range(num_modalities):
            start_ch = i * channels_per_modality
            end_ch = (i + 1) * channels_per_modality
            modality = layers.Lambda(lambda x: x[..., start_ch:end_ch])(inp)
            # Apply modality-specific attention
            modality = CBAM_3d(modality)
            modalities.append(modality)

        # Fuse modalities
        if len(modalities) == 2:
            fused = layers.concatenate(modalities)
        else:
            fused = layers.concatenate(modalities)
    else:
        fused = CBAM_3d(inp)

    # Encoder
    encoder_1 = encoder_block_3d(fused, f)
    encoder_2 = encoder_block_3d(encoder_1[0], f*2)
    encoder_3 = encoder_block_3d(encoder_2[0], f*4)
    encoder_4 = encoder_block_3d(encoder_3[0], f*8)

    # Bottleneck
    neck_in = layers.Conv3D(f*16, (3, 3, 3), activation='relu', padding='same',
                            name='Model_Neck_1', kernel_initializer='he_normal', kernel_regularizer='l1')(encoder_4[0])
    neck = layers.Conv3D(f*16, (3, 3, 3), activation='relu', padding='same',
                         name='Model_Neck_2', kernel_initializer='he_normal', kernel_regularizer='l1')(neck_in)
    neck = CBAM_3d(neck)  # Attention in bottleneck
    neck_out = neck

    # Decoder
    decoder_4 = decoder_block_3d(neck_out, encoder_4[1], f*8)
    decoder_3 = decoder_block_3d(decoder_4, encoder_3[1], f*4)
    decoder_2 = decoder_block_3d(decoder_3, encoder_2[1], f*2)
    decoder_1 = decoder_block_3d(decoder_2, encoder_1[1], f)

    # Output
    output = layers.Conv3DTranspose(1, (2, 2, 2), padding='same',
                                    kernel_initializer='he_normal', kernel_regularizer='l1')(decoder_1)
    out_image = layers.Activation('sigmoid')(output)
    model = keras.Model(inputs=inp, outputs=out_image)

    if not encoder_trainable:
        for layer in model.layers:
            if layer.name == 'Model_Neck_1':
                break
            layer.trainable = False

    return model


def build_multimodal_discriminator(image_shape, num_modalities=1):
    """
    Multi-modal discriminator for adversarial training
    """
    inp = layers.Input(image_shape)

    # Process each modality separately then combine
    channels_per_modality = image_shape[-1] // num_modalities
    modality_features = []

    for i in range(num_modalities):
        start_ch = i * channels_per_modality
        end_ch = (i + 1) * channels_per_modality
        modality = layers.Lambda(lambda x: x[..., start_ch:end_ch])(inp)

        # Modality-specific feature extraction
        x = layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same')(modality)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        modality_features.append(x)

    # Combine modalities
    if len(modality_features) > 1:
        combined = layers.concatenate(modality_features)
    else:
        combined = modality_features[0]

    # Continue with combined features
    x = layers.Conv3D(256, (4, 4, 4), strides=(2, 2, 2), padding='same')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv3D(512, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Output layer
    output = layers.Conv3D(1, (4, 4, 4), strides=(1, 1, 1), padding='same')(x)
    output = layers.Activation('sigmoid')(output)

    model = keras.Model(inputs=inp, outputs=output)
    return model
