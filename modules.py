import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from keras.models import Model
import keras.backend as K
from tensorflow import keras


class Yolo_Reshape(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        super(Yolo_Reshape, self).__init__()
        self.target_shape = tuple(target_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
        })
        return config

    def call(self, input):
        # grids 7x7
        S = [self.target_shape[0], self.target_shape[1]]
        # classes
        C = 2
        # no of bounding boxes per grid
        B = 1

        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B

        # class probabilities
        class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)

        # confidence
        confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)

        # boxes
        boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)

        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs


lrelu = LeakyReLU(alpha=0.1)

def block_1(inputs):
  conv = Conv2D(64, (7, 7), strides=(1,1), activation=lrelu, padding='same')(inputs)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_2(conv):
  conv = Conv2D(192, (3, 3), activation=lrelu, padding='same')(conv)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_3(conv):
  conv = Conv2D(128, (1, 1), activation=lrelu, padding='same')(conv)
  conv = Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(256, (1, 1), activation=lrelu, padding='same')(conv)
  conv = Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_4(conv):
  for i in range(4):
    conv = Conv2D(256, (1, 1), activation=lrelu, padding='same')(conv)
    conv = Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(512, (1, 1), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_5(conv):
  for i in range(2):
    conv = Conv2D(512, (1, 1), activation=lrelu, padding='same')(conv)
    conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), strides=(2,2), padding='same')(conv)
  print(conv.shape)
  return conv

def block_6(conv):
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  conv = Conv2D(1024, (3, 3), activation=lrelu, padding='same')(conv)
  print(conv.shape)
  return conv

def block_7(conv):
  conv = Flatten()(conv)
  conv = Dense(512)(conv)
  conv = Dense(1024)(conv)
  conv = Dropout(0.5)(conv)
  conv = Dense(1470, activation='sigmoid')(conv)
  print(conv.shape)
  output = Yolo_Reshape(target_shape=(7,7,30))(conv)
  print(output.shape)
  return output

inputs = Input(shape=(448,448,3))
conv = block_1(inputs)
conv = block_2(conv)
conv = block_3(conv)
conv = block_4(conv)
conv = block_5(conv)
conv = block_6(conv)
output = block_7(conv)


model = Model(inputs, output)



class CustomLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (20, 0.001),
    (40, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
