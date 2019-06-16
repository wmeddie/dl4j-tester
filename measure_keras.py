#!/usr/bin/env python

from time import monotonic
import math

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam

epochs = 3
batch_size = 256
num_classes = 10
inference_count = 1000000
zero = np.zeros((1, 28 * 28), dtype=np.float32)

def make_simple_model():
    model = Sequential()
    model.add(Dense(768, activation='relu', input_shape=(28 * 28,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.004, amsgrad=True),
                  metrics=['accuracy'])

    return model


def make_lenet_model():
    model = Sequential()
    model.add(Reshape(28, 28, 1))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=Adam(0.004, amsgrad=True),
                  metrics=['accuracy'])

    return model


def measure_training(model):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    start = monotonic()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    end = monotonic()

    return end - start


def measure_inference(model):
    inference_times = []

    for _ in range(100):
        model.predict(zero)
    for i in range(inference_count):
        start = monotonic()
        a = model.predict(zero)
        result = np.argmax(a)
        stop = monotonic()

        inference_times.append(stop - start)

        assert(result >= 0)

    return inference_times

def percentile(p, seq):
    assert min(seq) == seq[0] and max(seq) == seq[len(seq) - 1]

    k = int(math.ceil((len(seq) - 1) * (float(p) / 100.0)))
    return seq[k]

if __name__ == '__main__':
    model = make_simple_model()

    print("Using model:")
    print(model.summary())
    print("Measuring training performance...")

    training_time = measure_training(model)

    print("Training took %s seconds" % (str(training_time)))
    print("(%s per epoch)" % (str(training_time / epochs)))

    print("Measuring inference performance...")

    start = monotonic()
    inference_times = measure_inference(model)
    end = monotonic()

    inference_times.sort()

    inference_mean = sum(inference_times) / len(inference_times)
    inference_min = min(inference_times)
    inference_max = max(inference_times)
    inference_median = percentile(50, inference_times)
    one_nine = percentile(90.0, inference_times)
    two_nines = percentile(99.0, inference_times)
    three_nines = percentile(99.9, inference_times)
    four_nines = percentile(99.99, inference_times)
    five_nines = percentile(99.999, inference_times)
    six_nines = percentile(99.9999, inference_times)
    s2ms = 1000.0

    print("Min (%f ms per sample)" % (inference_min * s2ms))
    print("Mean (%f ms per sample)" % (inference_mean * s2ms))
    print("Median (%f ms per sample)" % (inference_median * s2ms))
    print("90 Percentile (%f ms per sample)" % (one_nine * s2ms))
    print("99 Percentile (%f ms per sample)" % (two_nines * s2ms))
    print("99.9 Percentile (%f ms per sample)" % (three_nines * s2ms))
    print("99.99 Percentile (%f ms per sample)" % (four_nines * s2ms))
    print("99.999 Percentile (%f ms per sample)" % (five_nines * s2ms))
    print("99.9999 Percentile (%f ms per sample)" % (six_nines * s2ms))
    print("Max (%f ms per sample)" % (inference_max * s2ms))
    print("Average %f inference per second" % (1 / inference_mean))
    print("Took %f seconds for %f inferences (%f inferences per second)" % (
        end - start,
        inference_count,
        float(inference_count) / (end - start)))

    out = open("results_keras.csv", "w")
    out.writelines(map(lambda x: str(x) + "\n", inference_times))
    out.close()
