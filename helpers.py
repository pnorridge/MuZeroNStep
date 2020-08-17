

from typing import Optional
import collections
import tensorflow as tf


MAXIMUM_FLOAT_VALUE = float('inf')


KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE 
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float): 
        self.maximum = max(self.maximum, value) 
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float: 
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum) 
        return value



class TFMinMaxStats(object):
    def __init__(self, shape):
        self.maximum = tf.Variable(tf.ones(shape)*-MAXIMUM_FLOAT_VALUE, trainable=False) 
        self.minimum = tf.Variable(tf.ones(shape)*MAXIMUM_FLOAT_VALUE, trainable=False) 

    def update(self, value: float): 
        value = tf.convert_to_tensor(value,dtype=tf.float32)
        min_s = tf.reduce_min(value, axis = 0)
        max_s = tf.reduce_max(value, axis = 0)
        self.minimum = tf.reduce_min([self.minimum, min_s], axis=0) 
        self.maximum = tf.reduce_max([self.maximum, max_s], axis=0) 
        
    def normalize(self, value: float) -> float: 
        if tf.reduce_all(self.maximum > self.minimum):
            return tf.divide(value - self.minimum,self.maximum - self.minimum) 
        else:
            return tf.convert_to_tensor(value,dtype=tf.float32)    


