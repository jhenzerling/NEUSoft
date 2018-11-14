#Learning Rate Object to handle updates and adjusting of Learning Rates (Derived in Network)
import os,sys,os.path
import importlib as i
import tensorflow as tf
from NEUnet import config_NEUnet as CONFIG
class LearningRater:
	#######################################################################################
	#######################################################################################
	#Initializer which sets the initial LR
	def __init__(self):
		#Sets Initial Learnrate, Learning Rate Decay, and Current Learning Rate, and update #
		self.LI = CONFIG.TRAINING['Learn Init']
		self.LR = CONFIG.TRAINING['Learn Init']
		self.LD = CONFIG.TRAINING['LR Decay']
		self.UN = CONFIG.TRAINING['LR Update']
			
	#######################################################################################
	#######################################################################################
	#Getters,Setters,Updaters
	def setLR(self,new):
		self.LR = new
	def setLD(self,new):
		self.LD = new
	def setLI(self,new):
		self.LI = new
	def setUN(self,new):
		self.UN = new

	def getLR(self):
		return self.LR
	def getLD(self):
		return self.LD
	def getLI(self):
		return self.LI
	def getUN(self):
		return self.UN

	#EVENTUALLY CAN ADD OTHER ADJUSTMENTS/SCHEDULING BUT THIS SUFFICES FOR NOW
	def UpdateLR(self,gstep,every,decay):
		self.LR = tf.train.exponential_decay(self.LI,gstep,every,self.LD,staircase=True)
		return self.LR
	def printLR(self):
		print('Learning Rate is now: ', self.LR.eval())
	
	#######################################################################################
	#######################################################################################


    
