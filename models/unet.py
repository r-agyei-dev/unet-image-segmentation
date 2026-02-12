# Import the necessary:
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Concatenate, Conv2DTranspose


# Convolution block
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# Encoder block
def encoding_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


# Decoder block
def decoding_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Build the U-Net model
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoding_block(inputs, 64)
    s2, p2 = encoding_block(p1, 128)
    s3, p3 = encoding_block(p2, 256)
    s4, p4 = encoding_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoding_block(b1, s4, 512)
    d2 = decoding_block(d1, s3, 256)
    d3 = decoding_block(d2, s2, 128)
    d4 = decoding_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

