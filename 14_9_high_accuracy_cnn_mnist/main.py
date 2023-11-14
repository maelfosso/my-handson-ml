# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import numpy as np


def build_model():
    tf.keras.backend.clear_session()
    tf.random.set_seed(170892)
    np.random.seed(42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=(28, 28, 1),
            kernel_initializer='he_normal'
        ),  # 28, 28, 32
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # (28-2)/2+1, (28-2)/2+1, 32 = 14, 14, 32
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal'
        ),  # 14, 14, 64
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # (14-2)/2+1, (14-2)/2+1, 64 = 7, 7, 64
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_initializer='he_normal'
        ),  # 7, 7, 128
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # (7-2)/2+1, (7-2)/2+1, 128 = 3, 3, 128
        tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(14, 14),
            activation='relu',
            kernel_initializer='he_normal'
        ),
        tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(1, 1),
            activation='softmax'
        ),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(
        #     units=10,
        #     activation='softmax',
        #     # kernel_initializer='he_normal'
        # ),  # 1, 1, 10
    ])

    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    print(model.summary())

    return model


def train(
        model: tf.keras.Sequential,
        train_set,
        valid_set
):
    x, y = train_set
    history = model.fit(
        x, y,
        validation_data=valid_set,
        epochs=10
    )

    return history


def run(train_set, valid_set):
    model = build_model()

    history = train(
        model,
        train_set,
        valid_set=valid_set
    )

    # return history.history


def main():
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full, x_test = x_train_full / 255., x_test / 255.
    x_train, x_valid = x_train_full[:-5000], x_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    x_train, x_valid, x_test = (tf.reshape(x_train, x_train.shape + (1,)),
                                tf.reshape(x_valid, x_valid.shape + (1,)),
                                tf.reshape(x_test, x_test.shape + (1,)))
    print(x_train.shape, x_valid.shape, x_test.shape)
    print(y_train.shape, y_valid.shape, y_test.shape)

    run(
        (x_train, y_train),
        (x_valid, y_valid)
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
