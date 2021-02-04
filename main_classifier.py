import tensorflow.keras as keras
from wrn import create_wide_residual_network

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

classifier_model = create_wide_residual_network((32, 32, 3), nb_classes=10, N=4, k=8)
classifier_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         optimizer="adam", metrics=["accuracy"])
early_stopping = keras.callbacks.EarlyStopping(patience=5,
                                               restore_best_weights=True)
history = classifier_model.fit(X_train_full, y_train_full, epochs=7,
                               validation_split=0.2,
                               callbacks=[early_stopping])
classifier_model.save_weights('models/class-1')
