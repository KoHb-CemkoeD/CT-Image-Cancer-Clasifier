import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, BatchNormalization, Dense, Dropout, multiply, GlobalAveragePooling2D, Lambda
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import preprocess_input, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6

EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
        EfficientNetB4, EfficientNetB5, EfficientNetB6]

current_folder = os.getcwd().replace('\\', '/')
working_dir = current_folder
dataset_path = current_folder + '/dataset/'
model_filename = 'CancerClassifierModel'
model_filepath = current_folder + '/models/' + model_filename
weights_filepath = current_folder + '/weights/' + model_filename

input_shape = (224, 224, 3)
num_class = 4


def create_model():
    ef = 0
    inp = Input(input_shape)

    # Base EfficientNet pretrained model
    base = EFNS[ef](input_shape=input_shape, weights="imagenet", include_top=False, drop_connect_rate=0.3)

    # variables for the attention mechanism and later
    pt_depth = base.layers[-1].output_shape[-1]
    pt_features = base(inp)
    bn_features = BatchNormalization()(pt_features)

    # Attention Mechanism
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding="same", activation="relu")(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding="same", activation="relu")(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding="same", activation="relu")(attn_layer)
    attn_layer = Conv2D(1, kernel_size=(1, 1), padding="valid", activation="sigmoid")(attn_layer)

    # Fan it out to all the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding="same", activation="linear", use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)

    # mask_features = multiply([Dropout(0.5)(bn_features), bn_features])
    # gap_features = GlobalAveragePooling2D()(mask_features)
    # gap_mask = GlobalAveragePooling2D()(Dropout(0.5)(bn_features))

    # To account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name="RescaleGAP")([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation="relu")(gap_dr))

    # Rebuild top
    x = Dense(4, activation="sigmoid")(dr_steps)

    model = Model(inputs=inp, outputs=x)
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Loss
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
    return model


def load_df():
    train_folder = dataset_path + "train"
    valid_folder = dataset_path + "valid"
    test_folder = dataset_path + "test"

    train_datagen = ImageDataGenerator(
        dtype='float32',
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )
    val_datagen = ImageDataGenerator(
        dtype='float32',
        preprocessing_function=preprocess_input,
    )
    test_datagen = ImageDataGenerator(
        dtype='float32',
        preprocessing_function=preprocess_input,
    )

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
    )

    test_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
    )
    validation_generator = val_datagen.flow_from_directory(
        valid_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
    )
    return train_datagen, val_datagen, test_datagen, train_generator, test_generator, validation_generator


def main():
    train_datagen, val_datagen, test_datagen, train_generator, \
        test_generator, validation_generator = load_df()

    model = create_model()

    # load_previews()

    # if os.path.isfile(model_filepath):
    #     print('load...')
    #     model.load_weights(model_filepath)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt1 = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = save_point()
    history = train_model(model, callbacks, test_generator, train_generator, validation_generator)

    model.save(model_filepath)
    model.save_weights(weights_filepath)

    make_report(model, test_generator)
    plot_results(history)


def load_previews():
    train_data = tf.keras.utils.image_dataset_from_directory(dataset_path + "train",
                                                             batch_size=16,
                                                             image_size=(512, 512),
                                                             seed=0,
                                                             label_mode='categorical',
                                                             shuffle=True)
    plt.figure(figsize=(20, 20))
    for img, labels in train_data.take(2):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img[i].numpy() / 255)
    plt.show()
    del train_data, img, labels, i, ax


def make_report(model, test_generator):
    # test = test_generator
    # score = model.evaluate(test, verbose=1)
    num_test_samples = len(test_generator)

    # num_classes = len(test_generator.class_indices)
    predicted_probabilities = model.predict(test_generator, steps=num_test_samples)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    true_labels = test_generator.classes

    report = classification_report(true_labels, predicted_labels)
    print(report)


def train_model(model, callbacks, test_generator, train_generator, validation_generator):
    epochs = 100
    history = model.fit(train_generator, callbacks=callbacks,
                        validation_data=validation_generator, epochs=epochs, verbose=1)
    return history


def save_point():
    # Defining a Checkpoint
    checkpoint = ModelCheckpoint(filepath=model_filepath,
                                 monitor='val_accuracy',
                                 mode='max',
                                 save_best_only=True,
                                 verbose=1,
                                 save_freq=5)
    # Defining a Early Stopping
    earlystop = EarlyStopping(monitor='val_accuracy',
                              min_delta=.5,
                              patience=5,
                              restore_best_weights=True)
    # Defining LR Reducing rate
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                  factor=0.1,
                                  patience=3,
                                  verbose=1,
                                  min_delta=0.8)
    # Putting the call backs in a callback list
    # callbacks = [checkpoint, earlystop, reduce_lr]
    callbacks = [checkpoint]
    return callbacks


def plot_results(history):
    # Plotting the accuracy charts
    history_dict = history.history

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)

    line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
    line2 = plt.plot(epochs, acc_values, label='Training Accuracy')

    plt.setp(line1, linewidth=1.8, marker='o', markersize=6.5)
    plt.setp(line2, linewidth=1.8, marker='*', markersize=8.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def gpu_configure(only_cpu=False):
    # Disable GPU support by setting the visible devices to only include the CPU
    if only_cpu:
        physical_devices = []
        tf.config.experimental.set_visible_devices(physical_devices, 'GPU')

    else:
        # Get the list of available physical devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    tf.config.LogicalDeviceConfiguration(
        memory_limit=None, experimental_priority=None, experimental_device_ordinal=None)

    gpu_configure(only_cpu=False)
    main()




# def create_model(input_shape, block1=True, block2=True, block3=True, block4=True, Dropout_ratio=0.25):
#     # * Create the model
#     model = Sequential()
#
#     # * configure the input shape
#     model.add(Input(shape=input_shape))
#
#     # * Add the first block
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
#                      trainable=block1))
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
#                      trainable=block1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(BatchNormalization())
#
#     # * Add the second block
#     model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
#                      trainable=block2))
#     model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
#                      trainable=block2))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(BatchNormalization())
#
#     # * Add the third block
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
#                      trainable=block3))
#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
#                      trainable=block3))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(BatchNormalization())
#     # * Add the fourth block
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#                      trainable=block4))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#                      trainable=block4))
#     model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
#                      trainable=block4))
#
#     # * flatten + Fc layer
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(Dropout_ratio))
#
#     # * Output layer
#     # model.add(Dense(3, activation='linear'))
#     model.add(Dense(4, activation='softmax'))
#     print('Done')
#     return model
