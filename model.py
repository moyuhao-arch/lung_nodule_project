import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow.keras.regularizers import l2

class ResNet2D:
  def _init_(self,input_shape=(32,32,1),num_classes=1,dropout_rate=0.5):
    self.input_shape = input_shape
    self.num_classes = num_classes
    self.dropout_rate = dropout_rate

  def conv_bn_relu(self,x,filters,kernel_size=(3,3),strides=(1.1),padding='same')
    x = layers.Conv2D(filters,kernel_size,strides=strides,padding=padding,kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

  def residual_block(self,x,filters,strides=(1,1)):
    shortcut = x
    x = self.conv_bn_relu(x,filters,strides=strides)
    x = layers.Conv2D(filters,(3,3),padding='same',kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if strides !=(1,1) or shortcut.shape[-1] != filters:
      shortcut = layers.Conv2D(filters,(1,1),strides=strides,padding='same')(shortcut)
      shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x,shortcut])
    x = layers.ReLU()(x)
    return x

  def build_resnet_backbone(self,input_tensor):
    x = input_tensor

    x = self.conv_bn_relu(x,32,kernel_size=(7,7),strides=(2,2))
    x = layers.MaxPooling2D((3,3),strides=(2,2),padding='same')(x)

 
