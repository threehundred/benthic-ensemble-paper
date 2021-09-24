import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50, InceptionResNetV2, VGG16, NASNetLarge, DenseNet201, DenseNet169, MobileNetV2

from keras_applications import densenet as dnet

from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalMaxPooling1D

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, MaxPooling2D
from tensorflow.keras import Input


def vgg(nb_classes, img_rows, img_cols, img_channels):
    model = VGG16(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet")

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    model = Model(input=model.input, output=dense_1)

    return model


def vgg_multipatch(nb_classes, img_height, img_width, img_channels):
    #model = VGG16(include_top=False, input_shape=(img_height, img_width, img_channels), classes=nb_classes,
    #                          weights="imagenet")

    input_1 = Input(shape=(img_height, img_width, img_channels))
    input_2 = Input(shape=(img_height, img_width, img_channels))
    #input_2 = MaxPooling2D((2, 2), strides=(2, 2))(input_2)
    model = ResNet50(include_top=False, input_tensor=input_1, classes=nb_classes, weights="imagenet")
    model2 = ResNet50(include_top=False, input_tensor=input_2, classes=nb_classes, weights="imagenet")

    #model.load_weights("/home/mat/Dev/keras-resnet/checkpoints-montara/weights-40.hdf5", by_name=True)
    #model2.load_weights("/home/mat/Dev/keras-resnet/checkpoints-montara/weights-40.hdf5", by_name=True)

    # add a global spatial average pooling layer
    x = model.output
    # x = Flatten()(x)
    x = Dense(2048, activation='relu', name='dense_1')(x)
    #x = Flatten()(x)
    model = Model(input=model.input, output=x)

    x2 = model2.output
    # x2 = Flatten()(x2)
    x2 = Dense(2048, activation='relu', name='dense_1')(x2)
    #x2 = Flatten()(x2)
    model2 = Model(input=model2.input, output=x2)

    for layer in model2.layers:
        layer.name += str("_two")

    joined = tensorflow.keras.layers.Concatenate()([model.output, model2.output])
    #joined = GlobalAveragePooling2D()(joined)
    #joined = Flatten()(joined)
    #joined = Dense(4096, activation='relu', name='hidden1')(joined)
    #joined = Dense(4096, activation='relu', name='hidden2')(joined)
    joined = Flatten()(joined)
    out = tensorflow.keras.layers.Dense(nb_classes, activation='softmax', name='dense_3')(joined)
    final_model = tensorflow.keras.models.Model(inputs=[model.input, model2.input], outputs=out)

    return final_model


def resnet(nb_classes, img_rows, img_cols, img_channels, custom_weights=None):
    model = ResNet50(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet")

    if custom_weights != None:
        model.load_weights(custom_weights, by_name=True)

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    model = Model(input=model.input, output=dense_1)

    return model


def resnet_multilabel(nb_classes, img_rows, img_cols, img_channels):
    model = ResNet50(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet")

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='sigmoid', name='dense_1')(x)

    model = Model(input=model.input, output=dense_1)

    return model


def resnet_multipatch(nb_classes, img_height, img_width, img_channels, custom_weights=None):
    input_1 = Input(shape=(img_height, img_width, img_channels))
    input_2 = Input(shape=(img_height, img_width, img_channels))
    input_3 = Input(shape=(img_height, img_width, img_channels))

    model = ResNet50(include_top=False, input_tensor=input_1, classes=nb_classes, weights="imagenet")
    model2 = ResNet50(include_top=False, input_tensor=input_2, classes=nb_classes, weights="imagenet")
    model3 = ResNet50(include_top=False, input_tensor=input_3, classes=nb_classes, weights="imagenet")

    if custom_weights != None:
        model.load_weights(custom_weights, by_name=True)
        model2.load_weights(custom_weights, by_name=True)
        model3.load_weights(custom_weights, by_name=True)

    # add a global spatial average pooling layer
    x = model.output
    # x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='relu', name='dense_1')(x)
    # dense_1 = Activation('relu')(x)
    model = Model(input=model.input, output=dense_1)

    x2 = model2.output
    # x2 = Flatten()(x2)
    dense_2 = Dense(nb_classes, activation='relu', name='dense_1')(x2)
    # dense_2 = Activation('relu')(x2)
    model2 = Model(input=model2.input, output=dense_2)

    x3 = model3.output
    # x2 = Flatten()(x2)
    dense_3 = Dense(nb_classes, activation='relu', name='dense_1')(x3)
    # dense_2 = Activation('relu')(x2)
    model3 = Model(input=model3.input, output=dense_3)

    for layer in model2.layers:
        layer.name += str("_two")

    for layer in model3.layers:
        layer.name += str("_three")

    joined = tensorflow.keras.layers.Add()([dense_1, dense_2, dense_3])  # equivalent to added = keras.layers.add([x1, x2])
    joined = Flatten()(joined)
    out = tensorflow.keras.layers.Dense(nb_classes, activation='softmax', name='dense_3')(joined)
    final_model = tensorflow.keras.models.Model(inputs=[input_1, input_2, input_3], outputs=out)

    return final_model


def inception_resnet(nb_classes, img_rows, img_cols, img_channels):
    model = InceptionResNetV2(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet")

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    model = Model(input=model.input, output=dense_1)

    return model


def nasnet(nb_classes, img_rows, img_cols, img_channels):
    model = NASNetLarge(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet")

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    model = Model(input=model.input, output=dense_1)

    return model


def mobilenetV2(nb_classes, img_rows, img_cols, img_channels):
    model = MobileNetV2(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet")

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='softmax', use_bias=True, name='dense_1')(x)

    model = Model(input=model.input, output=dense_1)

    return model



def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """

    # assuming channels last
    bn_axis = 3 if dnet.backend.image_data_format() == 'channels_last' else 1
    x1 = dnet.layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = dnet.layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = dnet.layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = dnet.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = dnet.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = dnet.layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)

    #x1 = dnet.layers.Conv2D(4, (1,1),
    #                        activation='relu',
    #                        name=name + 'reducer')(x1)

    x1 = Dropout(0.2)(x1, training=True)

    print("---------------------------------> patched")

    x = dnet.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

dnet.conv_block = conv_block

def bayesian_densnet(nb_classes, img_rows, img_cols, img_channels, custom_weights=None):
    #dnet.conv_block = conv_block

    densenet(nb_classes, img_rows, img_cols, img_channels, custom_weights)

def densenet(nb_classes, img_rows, img_cols, img_channels, custom_weights=None):
    model = DenseNet169(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights=None, pooling='avg')

    if custom_weights != None and custom_weights != 'imagenet':
        model.load_weights(custom_weights, by_name=True)

    # add a global spatial average pooling layer
    x = model.output
    #x = Flatten()(x)
    dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    model = Model(inputs=model.input, outputs=dense_1)

    return model

def densenet_shrunk(nb_classes, img_rows, img_cols, img_channels, custom_weights=None):
    model = DenseNet169(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights=None, pooling=None)

    if custom_weights != None:
        model.load_weights(custom_weights, by_name=True)

    # add a global spatial average pooling layer
    x = model.output
    #x = Flatten()(x)
    x = GlobalMaxPooling2D(name='avg_pool2')(x)
    x = GlobalMaxPooling2D(name='avg_pool3')(x)


    dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    model = Model(inputs=model.input, outputs=dense_1)

    return model

def densenet_extra_params(nb_classes, img_rows, img_cols, img_channels):
    model = DenseNet169(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet", pooling='avg')

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    #dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    extra_params_model = Sequential()
    #extra_params_model.add(Dense(6, input_shape=(3,), activation='relu', name='dense_params'))
    extra_params_model.add(Dense(3, input_shape=(3,), name='dense_params'))
    #extra_params_model_output = Flatten()(extra_params_model)

    joined = tensorflow.keras.layers.Concatenate()([x, extra_params_model.output])
    #joined = Flatten()(joined)

    out = tensorflow.keras.layers.Dense(nb_classes, activation='softmax', name='dense_3')(joined)
    final_model = tensorflow.keras.models.Model(inputs=[model.input, extra_params_model.input], outputs=out)

    return final_model

def densenet_bioregions(nb_classes, img_rows, img_cols, img_channels):
    model = DenseNet169(include_top=False, input_shape=(img_rows, img_cols, img_channels), classes=nb_classes,
                              weights="imagenet", pooling='avg')

    # add a global spatial average pooling layer
    x = model.output
    x = Flatten()(x)
    #dense_1 = Dense(nb_classes, activation='softmax', name='dense_1')(x)

    extra_params_model = Sequential()
    #extra_params_model.add(Dense(6, input_shape=(3,), activation='relu', name='dense_params'))
    extra_params_model.add(Dense(31, input_shape=(31,), name='dense_params'))
    #extra_params_model_output = Flatten()(extra_params_model)

    joined = tensorflow.keras.layers.Concatenate()([x, extra_params_model.output])
    #joined = Flatten()(joined)

    out = tensorflow.keras.layers.Dense(nb_classes, activation='softmax', name='dense_3')(joined)
    final_model = tensorflow.keras.models.Model(inputs=[model.input, extra_params_model.input], outputs=out)

    return final_model

def densenet_multipatch(nb_classes, img_height, img_width, img_channels, custom_weights=None):

    input_1 = Input(shape=(img_height, img_width, img_channels))
    input_2 = Input(shape=(img_height, img_width, img_channels))
    # input_2 = MaxPooling2D((2, 2), strides=(2, 2))(input_2)
    model = DenseNet169(include_top=False, input_tensor=input_1, classes=nb_classes, weights="imagenet", pooling='avg')
    model2 = DenseNet169(include_top=False, input_tensor=input_2, classes=nb_classes, weights="imagenet", pooling='avg')

    if custom_weights != None:
        model.load_weights(custom_weights, by_name=True)
        model2.load_weights(custom_weights, by_name=True)

    # add a global spatial average pooling layer
    x = model.output
    # x = Flatten()(x)
    x = Dense(2048, activation='relu', name='dense_1')(x)
    # x = Flatten()(x)
    model = Model(input=model.input, output=x)

    x2 = model2.output
    # x2 = Flatten()(x2)
    x2 = Dense(2048, activation='relu', name='dense_1')(x2)
    # x2 = Flatten()(x2)
    model2 = Model(input=model2.input, output=x2)

    for layer in model2.layers:
        layer.name += str("_two")

    joined = tensorflow.keras.layers.Concatenate()([model.output, model2.output])
    # joined = GlobalAveragePooling2D()(joined)
    # joined = Flatten()(joined)
    # joined = Dense(4096, activation='relu', name='hidden1')(joined)
    # joined = Dense(4096, activation='relu', name='hidden2')(joined)
    joined = Flatten()(joined)
    out = tensorflow.keras.layers.Dense(nb_classes, activation='softmax', name='dense_3')(joined)
    final_model = tensorflow.keras.models.Model(inputs=[model.input, model2.input], outputs=out)

    return final_model