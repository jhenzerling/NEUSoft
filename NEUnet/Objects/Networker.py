#Network Object to Put Network with Data
import os,sys,os.path
import importlib as i
import tensorflow as tf
#Configuration Files
from NEUnet import config_NEUnet as CONFIG
from NEUnet.Objects import LearningRater as L

#Sets things up for the network
#NOT WORKING FOR RESNET YET - ALSO NEED TO EDIT NETWORKS TO REFER TO THE PROPER CONFIG FILE!!!!
#NEXT STEP IS TO FINEGLE THE TRAINING STEPS
class Network:
	#######################################################################################
	#######################################################################################
	#Initializer by checking if Network is Present then Importing
	def __init__(self,dataobject):
		self.LR = L.LearningRater()
		self.data = dataobject
		archi = CONFIG.MAIN['Archi']
		name = CONFIG.MAIN['Data']
		path = self.data.pobj.getAPath(archi)
		netmodloc = 'Networks.' + archi
		#Check for Network
		if(os.path.exists(path)):
   			print(archi + ' Network Present.')
		else:
    			print(archi + ' Network Not Present.')
    			quit()
		try:
			self.Nmod = i.import_module('.' + archi, package = netmodloc)
		except ImportError:
			sys.exit('Failed to import, net name fail: ' + netmodloc)	
	#Setters
	def setTensors(self,data):
		[hsize,vsize,colours,self.cnumb] 	= data.getImageInfo()
		[self.rawinput,self.x,self.y_] 		= data.getPlaceholders()
	#Set up the network (i.e.) build w/placeholders or initialize via ssnet
	#Add to NETCONFIG how you prepare the network so it can be called right
	def setNetwork(self):
		self.net,self.ep = self.Nmod.build(
				self.x,self.cnumb,CONFIG.MAIN['Train'],CONFIG.MAIN['Debug'])
	#Set all since requires a certain order
	def setCost(self):
		self.learnrate = self.LR.getLI()
		self.keep_prob 	= tf.placeholder(tf.float32, name = 'keep_prob')
		[self.costfunction,
		self.train_step,
		self.accuracy] = self.Generic_Cost_Setup(self.y_,self.net,self.learnrate)
	#Perhaps need a Cost object/setting somewhere so it's easier to edit CONSIDER + a name_Coster
	def Generic_Cost_Setup(self,y,network,LR):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=network))
   		train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)   
    		correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y, 1))
   		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
		return [cross_entropy, train_step, accuracy]
	def UpdateLR(self,gstep,every,decay):
		self.learningrate = self.LR.UpdateLR(gstep,every,decay)
	def printLR(self):
		self.LR.printLR()
	def setHistograms(self,name):
		for var in tf.trainable_variables():
        		tf.summary.histogram(var.name,var,collections=[name])
	def setAll(self,data):
		self.setTensors(data)
		self.setNetwork()
		self.setCost()
		self.setHistograms('my_summ')
	#######################################################################################
	#######################################################################################
	#Getters
	def getNetwork(self):
		return self.net
	def getEndPoints(self):
		return self.ep
	def getIOT(self):
		return [self.x,self.y_]
	def getF(self):
		return self.rawinput
	def getFlat(self):
		return self.rawinput
	def getCostT(self):
		return [self.keep_prob,self.costfunction,self.accuracy]
	def getTrainStep(self):
		return self.train_step

	#######################################################################################
	#######################################################################################


    
