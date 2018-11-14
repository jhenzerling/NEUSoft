#MNIST Data Module
#Configuration Files
from NEUnet import config_NEUnet as M
from Modules.MNIST import config_MNIST as C

#Type of problem being solved and way data is stored and name
variety 	= C.DATA['Variety']
process		= C.DATA['Process']
name		= M.MAIN['Data']
  
#Pull data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Modules/MNIST/Data/', one_hot=True)

#Image Dimensions and Colours and Classes and Axes
hsize 		= C.DATA['H']
vsize 		= C.DATA['V']
colours 	= C.DATA['Colours']
cnumb 		= C.DATA['Classes']
#Size of the dataset
trainsize 	= C.DATA['Training Set Size']
testsize 	= C.DATA['Testing Set Size']
#Batchsize designates number of points in each separate chunk
#Stepnumber designates number batches to go over
trbatchsize 	= M.TRAINING['Training Batch Size']
tebatchsize	= M.TRAINING['Testing Batch Size']

def batcher():
	t1,t2		= mnist.train.next_batch(trbatchsize)
	tr 		= [t1,t2]
	te 		= [mnist.test.images,mnist.test.labels]
	result 		= [tr,te]
	return result

##################################################################

		



    
