#CIFAR10 HANDLING
#Libraries for dealing with CIFAR
import cPickle,os
import numpy 		as np
import tensorflow 	as tf
#Configuration Files
from NEUnet import config_NEUnet as M
from Modules.CIFAR10 import config_CIFAR10 	as C

#Type of problem being solved and way data is stored and name
variety 	= C.DATA['Variety']
process		= C.DATA['Process']
name		= M.MAIN['Data']
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

#Functions for Processing
###########################################################################################

#Loading CIFAR10 Data from File
#Data is taken in as COLOUR,HSIZE,VSIZE, need to rearrange later
def dataloader(setnumb):
    
    #Path and selecting batch
    path = './Modules/CIFAR10/Data/'
    
    if (setnumb == 0):
        file = 'test_batch'
    else:
        file = 'data_batch_' + str(setnumb)
        
    f 	   = open(path+file, 'rb')
    dict   = cPickle.load(f)
    images = dict['data']
    labels = dict['labels']
   
    #Assign Image and Label to Flattened Arrays
    imagearray = np.array(images)
    labelarray = np.array(labels)   #   (10000,)
    return [imagearray, labelarray]

#Formats inputted data into a usable way
def setmaker(imagearray, labelarray, setsize):
    #Trainingsize => TrainBatchSize
    setimage = np.empty((setsize,hsize*vsize*colours)) #image
    setlabel = np.zeros((setsize,cnumb)) #labels => 1hotvector

    i = 0
    while (i < setsize):               
        setimage[i,:] 		  = imagearray[i,:]
        setlabel[i,labelarray[i]] = 1
        i+=1

    setimage = np.reshape(setimage, [setsize,colours,hsize,vsize]) 
    setimage = np.transpose(setimage, [0,2,3,1])
    setlabel = np.reshape(setlabel, [setsize,cnumb])
    setimage = setimage/255.
    return [setimage, setlabel]

#Using prior function, loads in the entire CIFAR10 dataset
def fulldata(pick,asize):
    size = asize
    if (pick == 1):
        d1 = setmaker(dataloader(1)[0],dataloader(1)[1],size)
        d2 = setmaker(dataloader(2)[0],dataloader(2)[1],size)
        d3 = setmaker(dataloader(3)[0],dataloader(3)[1],size)
        d4 = setmaker(dataloader(4)[0],dataloader(4)[1],size)
        d5 = setmaker(dataloader(5)[0],dataloader(5)[1],size)

        compimage = np.concatenate([d1[0],d2[0],d3[0],d4[0],d5[0]], 0)
        complabel = np.concatenate([d1[1],d2[1],d3[1],d4[1],d5[1]], 0)

    elif (pick == 0):
        d0 = setmaker(dataloader(0)[0],dataloader(0)[1],size)

        compimage = d0[0]
        complabel = d0[1]
    else:
        print('Use 0 for train, 1 for test. Terminating.')
    return [compimage,complabel]


###########################################################################################

#Load in the training/testing data and assign to sets
with tf.variable_scope('Cifar10_Data'):
    traincifarsets  = fulldata(1,10000)
    testcifarsets   = fulldata(0,10000)

    #Assign for use in network
    g1,g2 = tf.train.shuffle_batch([traincifarsets[0],traincifarsets[1]], 
		trbatchsize, num_threads=1, capacity = 50000, enqueue_many=True, 
		allow_smaller_final_batch = False, min_after_dequeue = 10000)
    h1,h2 = tf.train.shuffle_batch([testcifarsets[0],testcifarsets[1]], 
		tebatchsize, num_threads=1, capacity = 50000, enqueue_many=True, 
		allow_smaller_final_batch = False, min_after_dequeue = 10000)

    trainbatch 	= [g1,g2]
    testbatch 	= [h1,h2]

    result = [trainbatch,testbatch]

def batcher():
    traincifarsets  = fulldata(1,10000)
    testcifarsets   = fulldata(0,10000)

    #Assign for use in network
    g1,g2 = tf.train.shuffle_batch([traincifarsets[0],traincifarsets[1]], 
		trbatchsize, num_threads=1, capacity = 50000, enqueue_many=True, 
		allow_smaller_final_batch = False, min_after_dequeue = 10000)
    h1,h2 = tf.train.shuffle_batch([testcifarsets[0],testcifarsets[1]], 
		tebatchsize, num_threads=1, capacity = 50000, enqueue_many=True, 
		allow_smaller_final_batch = False, min_after_dequeue = 10000)
    trainbatch 	= [g1,g2]
    testbatch 	= [h1,h2]
    result = [trainbatch,testbatch]
    return result

###########################################################################################




