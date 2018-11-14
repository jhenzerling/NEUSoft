#This module groups up the learning techniques so it's easier
#File Handling
import os
#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from NEUnet import config_NEUnet as CM
from Modules.CIFAR10 import Loader_CIFAR10 as L
#Tensorflow Libraries
import tensorflow as tf

def setConfig():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return config

def setThreads():
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)
	return [coord,threads]

def endThreads(coord,threads):
	coord.request_stop()
        coord.join(threads)	

def setSummaries(sess,costfunction,accuracy,trainbatch,number):
	tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	tf.summary.scalar("Cost Function", costfunction)
        tf.summary.scalar("Accuracy", accuracy)
	trim  = tf.reshape(trainbatch[0],[CM.TRAINING['Training Batch Size'],32,32,3])
	tf.summary.image('train_input',trim,number)

def setWriters(sess):
	merged = tf.summary.merge_all()
	tf.trainable_variables(tf.GraphKeys.GLOBAL_VARIABLES)
        writertrain=tf.summary.FileWriter('Output/Graphs/Train', sess.graph)
        writertrain.add_graph(sess.graph)
        writertest=tf.summary.FileWriter('Output/Graphs/Test', sess.graph)
        writertest.add_graph(sess.graph)
        saver = tf.train.Saver()
	return [writertrain,writertest,merged,saver]

def setFeeds(a,b,c,d,kp,sess):
	Qtestbatch0,Qtestbatch1 = sess.run([a,b])
	testfeed = {c: Qtestbatch0, d: Qtestbatch1, kp: 1.0}
	return testfeed

def FeedSummaries(merge,accu,sess,trainfeed,testfeed):
	summarytrain,acctrain = sess.run([merge,accu], feed_dict=trainfeed)
	summarytest,acctest = sess.run([merge,accu], feed_dict=testfeed)
	return [summarytrain,acctrain,summarytest,acctest]

def AddSummaries(writertrain,writertest,summarytrain,summarytest,i):
	writertrain.add_summary(summarytrain,i)
	writertest.add_summary(summarytest,i)

def feedData(sess,trainfeed,testfeed,accuracy,costfunction):
	train_accuracy 		= accuracy.eval(session=sess,feed_dict=trainfeed)
        test_accuracy 		= accuracy.eval(feed_dict=testfeed)
        train_loss 		= costfunction.eval(feed_dict=trainfeed)
        test_loss 		= costfunction.eval(feed_dict=testfeed)

	return [train_accuracy,test_accuracy,train_loss,test_loss]

def outputprint(step,tracc,teacc,trloss,teloss,i):
    print('Progress: (%g) %%, \t\t\t' % (i*100/step))
    print('Training Accuracy: (%g), \t\t\t Training Loss: (%g)' % (tracc, trloss))
    print('Testing Accuracy: (%g), \t\t\t Testing Loss: (%g)' % (teacc, teloss))
	
#Takes in Objects
def Training(dobj,nobj):
	f					= nobj.getF()
	[x,y]					= nobj.getIOT()
	[trainbatch,testbatch]			= dobj.getResult()
	[keep_prob,costfunction,accuracy] 	= nobj.getCostT()
	train_step 				= nobj.getTrainStep()
	graphtrigger				= CM.MAIN['Graph']
	stepnumber				= CM.TRAINING['Step']
	lognumber				= CM.MAIN['Logging']
	saverate				= CM.TRAINING['Save']
	lrdate					= CM.TRAINING['LR Update']

	config = setConfig()
	with tf.Session(config=config) as sess:
		[coords,threads] = setThreads()
		with tf.variable_scope('Session'):
			init = tf.global_variables_initializer()
			sess.run(init)
		if graphtrigger == True:
			setSummaries(sess,costfunction,accuracy,trainbatch,10)
			[writertrain,writertest,merged,saver] = setWriters(sess)
			for i in range(stepnumber):
				with tf.variable_scope('Feeding'):
					trainfeed = setFeeds(trainbatch[0],trainbatch[1],x,y,keep_prob,sess)
					testfeed = setFeeds(testbatch[0],testbatch[1],x,y,keep_prob,sess)
					if i % (stepnumber/lognumber) == 0:
						[summarytrain,acctrain,summarytest,acctest] = FeedSummaries(merged,accuracy,sess,trainfeed,testfeed)
						AddSummaries(writertrain,writertest,summarytrain,summarytest,i)
						[train_accuracy,test_accuracy,train_loss,test_loss] = feedData(sess,trainfeed,testfeed,accuracy,costfunction)
						outputprint(stepnumber,train_accuracy,test_accuracy,train_loss,test_loss,i)
				with tf.variable_scope('Training'):
					train_step.run(feed_dict=trainfeed)
					if i != 0:
						if i % (stepnumber/lrdate) == 0:
							nobj.UpdateLR(i,(stepnumber/lrdate),.75)
							nobj.printLR()
				if i % saverate == 0:
					ssf_path = saver.save(sess, 'Output/Weights/CIFAR10',global_step=i)
			endThreads(coords,threads)


