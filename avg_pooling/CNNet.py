import tensorflow as tf 

def avg_pool(x):
    '''
    全局平均池化层，使用一个与原有输入同样尺寸的filter进行池化，'SAME'填充方式  池化层后
         out_height = in_hight / strides_height（向上取整）
         out_width = in_width / strides_width（向上取整）
    
    args；
        x:输入图像 形状为[batch,in_height,in_width,in_channels] 
    '''
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def batch_norm_layer(value,is_training=False,name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果
    
    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training is True:
        #训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)

def conv(x, d_out, name, is_training = False):
    d_in = x.get_shape()[-1].value
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        # 创建或获取名叫weights的变量
        weights = tf.get_variable('weights', shape=[5, 5, d_in, d_out], 
                                                dtype=tf.float32,
                                                initializer = tf.truncated_normal_initializer(stddev = 0.1))
        # 创建或获取名叫biases的变量
        biases = tf.get_variable('biases', shape=[d_out], initializer = tf.constant_initializer(0.1))
    
        conv = tf.nn.conv2d(x, weights, strides = [1, 1, 1, 1], padding = 'SAME')
        bn_result = batch_norm_layer(conv + biases, is_training=is_training)
        return tf.nn.relu(bn_result)

# 池化层操作定义
def maxpool(input, name):
    pool = tf.nn.max_pool(input, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
    # 局部响应归一化
    # norm = tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    return pool

def Layer(x, n_out, name):
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        weight = tf.get_variable('weights', shape=[n_in, n_out], 
                                        dtype=tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable('biases', shape=[n_out], initializer = tf.constant_initializer(0.1))
        
        activation = tf.nn.relu_layer(x, weight, biases)
        # print_layer(activation)
        return activation



def inference(images, keep_prob, n_cls, isTraining = False):
    
    conv1 = conv(images, 32, 'conv1', isTraining)
    pool1 = maxpool(conv1, 'pool1')
    
    conv2 = conv(pool1, 64, 'conv2', isTraining)
    pool2 = maxpool(conv2, 'pool2')
    
    conv3 = conv(pool2, 128, 'conv3', isTraining)
    pool3 = maxpool(conv3, 'pool3')

    conv4 = conv(pool3, 256, 'conv4', isTraining)
    pool4 = maxpool(conv4, 'pool4')

    conv5 = conv(pool4, 512, 'conv5', isTraining)
    pool5 = maxpool(conv5, 'pool5')

    
    conv6 = conv(pool5, n_cls, 'conv6', isTraining)
    nt_pool6 = avg_pool(conv6)
    nt_pool6_flat = tf.reshape(nt_pool6, [-1, 10])

    return nt_pool6_flat
    '''
    flatten = tf.reshape(pool5, [-1, 2*2*512])
    layer1 = Layer(flatten, 5120, 'layer1')
    dropout1 = tf.nn.dropout(layer1, keep_prob)

    layer2 = Layer(dropout1, 5120, 'layer2')
    dropout2 = tf.nn.dropout(layer2, keep_prob)

    with tf.variable_scope('softmax_linear', reuse = tf.AUTO_REUSE):
        n_in = dropout2.get_shape()[-1].value
        weights = tf.get_variable('weights', shape=[n_in, n_cls], 
                                        dtype=tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev = 0.05))

        biases = tf.get_variable('biases', shape=[n_cls], initializer = tf.constant_initializer(0.1))

        softmax_linear = tf.add(tf.matmul(dropout2, weights), biases, name = 'output')

    return softmax_linear
    '''
    '''
    layer3 = Layer(dropout2, n_cls, 'layer3')
    return layer3
    '''