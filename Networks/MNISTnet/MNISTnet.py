#An MNIST-specific network architecture
import tensorflow.contrib.slim as slim
import tensorflow.python.platform
import tensorflow as tf

#BUILDING THE NETWORK

def build(input_tensor, num_class, trainable, debug):

    outputs_collections = "MNISTnet"
    #Take in input data as 'net' object
    #Expect an input of [BatchSize, Channels, HSize, VSize]
    net = input_tensor
    #net = [tf.reshape(net,[trbatchsize,hsize,vsize,colours])]
    if debug: print 'Input Tensor: ', input_tensor.shape

    #Define the number of filters to convolve
    filters 	= 32

    with tf.variable_scope('Network'):
	#net = tf.layers.dropout(net, idropout)
	net = slim.conv2d(inputs      			= net,       # input tensor
                                  num_outputs 		= filters,   # number of filters/feature maps
                                  kernel_size 		= [5,5],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer 	= tf.truncated_normal_initializer(stddev=0.1),
                                  biases_initializer 	= tf.constant_initializer(0.1))
	 
	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	
	print 'After Conv1 and MaxPool1 ', net.shape

	net = slim.conv2d(inputs   			= net,       # input tensor
                                  num_outputs 		= filters*2,   # number of filters/feature maps
                                  kernel_size		= [5,5],     # kernel size
                                  stride      		= 1,         # stride size
                                  trainable   		= trainable, # train or inference
                                  activation_fn 	= tf.nn.relu,
                                  weights_initializer 	= tf.truncated_normal_initializer(stddev=0.1),
                                  biases_initializer 	= tf.constant_initializer(0.1))

	net = slim.max_pool2d(inputs      		= net,    # input tensor
                              kernel_size 		= [2,2],  # kernel size
                              stride      		= 2,
			      padding	  		= 'SAME')
	
	print 'After Conv2 and MaxPool2 ', net.shape


	#Flatten the network to 1D
        net = slim.flatten(net, scope='Flatten_Step')
        if debug: print 'After flattening', net.shape              
        net = slim.fully_connected(net, 1024)  
        if debug: print 'After Fully Connected Layer ', net.shape
        if trainable:
        	net = slim.dropout(net, keep_prob=.5, is_training=trainable, scope='fc_dropout')
        #Set through a final fc layer
        net = slim.fully_connected(net, int(num_class), scope='FC_Final')
        if debug: print 'After Fully Connected Layer Final', net.shape

        end_points = slim.utils.convert_collection_to_dict(outputs_collections)
    
    #Send back the network
    return net,end_points



