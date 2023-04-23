import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.utils.vis_utils import plot_model  # necesita matplotlib, pydot, graphviz
from tensorflow.keras.callbacks import TensorBoard

%load_ext tensorboard
%tensorboard --logdir logs

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada
x_train = x_train / 255.0
x_test = x_test / 255.0

# Crear el modelo
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(150, activation="relu"),
    tf.keras.layers.Dense(10, activation ="softmax") # Una neurona porque la salida es binaria
])


modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),    
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
])

# Compilar el modelo
modeloDenso.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

modeloCNN.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


# Entrenar el modelo
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
tensorboardCNN = TensorBoard(log_dir = "logs/CNN")
modeloCNN.fit(x_train, y_train, validation_split = 0.15, epochs=10, callbacks = [tensorboardCNN])

tensorboardDenso = TensorBoard(log_dir = "logs/Denso")
modeloDenso.fit(x_train, y_train, validation_split = 0.15, epochs=10, callbacks = [tensorboardCNN])

# Evaluar el modelo en el conjunto de prueba
#model.evaluate(x_test, y_test)

# Exportar el modelo
modeloDenso.save('modelo_Denso_mnist.h5')
modeloCNN.save('modelo_CNN_mnist.h5')