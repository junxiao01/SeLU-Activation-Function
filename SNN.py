from keras import backend as K
from keras.layers.core import Lambda
from keras.engine import Layer
import tensorflow as tf

class SeLU(Layer):
    """
    define SELU function 
    """
    def __init__(self,alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946,
                 **kwargs):
    	self.alpha = alpha
    	self.scale = scale
    	super(SeLU,self).__init__(**kwargs)
    def call(self,x,mask=None):
        return self.scale*tf.where(x>=0.0, x, self.alpha*tf.nn.elu(x))
    def get_output_shape_for(self, input_shape):
        #we don't change the input shape
        return self.compute_output_shape(input_shape)


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import numbers

class Dropout_SeLU(Layer):
    """
    define corresponding dropout function
    """
    def __init__(self, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False,**kwargs):
        self.rate = rate
        self.alpha = alpha
        self.fixedPointMean = fixedPointMean
        self.fixedPointVar = fixedPointVar
        self.noise_shape = noise_shape
        self.seed = seed
        self.name = name
        self.training = training
        super(Dropout_SeLU,self).__init__(**kwargs)
    def call(self,x,mask=None):
        keep_prob = 1.0 - self.rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        alpha = ops.convert_to_tensor(self.alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        if tensor_util.constant_value(keep_prob) == 1:
            return x
        noise_shape = self.noise_shape if self.noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=self.seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)
        a = tf.sqrt(self.fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-self.fixedPointMean,2) + self.fixedPointVar)))
        b = self.fixedPointMean - a * (keep_prob * self.fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)