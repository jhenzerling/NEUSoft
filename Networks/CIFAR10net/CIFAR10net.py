#An CIFAR10-specific network architecture
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#BUILDING THE NETWORK

def build(input_tensor, num_class, trainable, debug):

    outputs_collections = "CIFAR10net"
    net = input_tensor
    if debug: print 'Input Tensor: ', input_tensor.shape
    #net = slim.dropout(net, keep_prob = .1, is_training=trainable,scope='dropout0')
    #Define the number of filters to convolve
    filters 	= 32

    with tf.variable_scope('Network'):
	#net = tf.layers.dropout(net, idropout)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	#net = slim.dropout(net, keep_prob = .2, is_training=trainable,scope='dropout1')
	print 'After Conv1 and MaxPool1 ', net.shape
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*2,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*2,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	#net = slim.dropout(net, keep_prob = .3, is_training=trainable,scope='dropout2')
	print 'After Conv2 and MaxPool2 ', net.shape
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*4,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*4,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	#net = slim.dropout(net, keep_prob = .4, is_training=trainable,scope='dropout3')
	print 'After Conv3 and MaxPool3 ', net.shape
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*8,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*8,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	#net = slim.dropout(net, keep_prob = .4, is_training=trainable,scope='dropout4')
	print 'After Conv4 and MaxPool4 ', net.shape
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*16,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters*16,   # number of filters/feature maps
                                  kernel_size 		= [3,3],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer = tf.contrib.layers.xavier_initializer(),
                                  biases_initializer 	= tf.zeros_initializer())
	#net = tf.layers.batch_normalization(net)
	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	#net = slim.dropout(net, keep_prob = .4, is_training=trainable,scope='dropout5')
	print 'After Conv5 and MaxPool5 ', net.shape
	#Flatten the network to 1D
        net = slim.flatten(net, scope='Flatten_Step')
        if debug: print 'After flattening', net.shape              
        net = slim.fully_connected(net, 512)  
        if debug: print 'After Fully Connected Layer ', net.shape
        if trainable:
        	net = slim.dropout(net, keep_prob=.5, is_training=trainable, scope='fc_dropout')
        #Set through a final fc layer
        net = slim.fully_connected(net, int(num_class), scope='FC_Final')
        if debug: print 'After Fully Connected Layer Final', net.shape

        end_points = slim.utils.convert_collection_to_dict(outputs_collections)
    
    #Send back the network
    return net,end_points



