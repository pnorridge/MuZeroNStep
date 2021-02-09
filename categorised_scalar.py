import tensorflow as tf

eps = 0.01

N = 600
offset = -300
basis = tf.transpose([tf.cast(tf.range(offset,offset+N),tf.float32)])

def normalise_scalars(x: tf.Tensor, eps: float = 0.01):
    #return tf.sign(x)*(tf.sqrt(tf.abs(x)+1) - 1 + eps*x)
    return tf.sign(x)*(tf.sqrt(tf.abs(x)+1) - 1)

def inverse_normalisation(x: tf.Tensor, eps: float = 0.01):
    # full inverse is y = (tf.sqrt(1 + 4*eps*(tf.abs(x)+1+eps))-1)/2/eps
    # return tf.sign(x)*(tf.pow(y,2.)-1)
    return tf.sign(x)*(tf.pow(tf.abs(x)+1,2.) - 1)

def categorise(x: tf.Tensor):
    x = tf.cast(normalise_scalars(x) - offset, tf.float32)
    ind = tf.cast(x,tf.int32)
    x = x-tf.cast(ind,tf.float32)

    tmp = tf.scatter_nd(tf.transpose([tf.range(0,len(ind)),ind]),1-x,[len(ind),N]) + \
    tf.scatter_nd(tf.transpose([tf.range(0,len(ind)),ind+1]),x,[len(ind),N]) 

    return tmp

def decategorise(x: tf.Tensor):
    x=tf.cast(x,tf.float32)
    return inverse_normalisation(tf.matmul(x,basis)) 

def categorised_initialiser(shape, dtype=tf.float32):
    d0 = shape.shape[0]
    return tf.scatter_nd(tf.transpose([tf.range(0,d0),N/2.]),tf.cast(1.,dtype),[d0,shape.shape[1]])
