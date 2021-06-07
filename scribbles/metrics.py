import tensorflow.keras.backend as K

from tensorflow.keras.metrics import mean_squared_error as K_mean_squared_error


def nmse(y_test, y_pred):

    return K_mean_squared_error(y_test, y_pred) / K.var(y_test)
