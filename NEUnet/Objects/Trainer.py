#This module groups up the learning techniques so it's easier
import sys,os
import importlib as i
#Main Configuration
from NEUnet import config_NEUnet as CONFIG
#Object that handles the training steps whatever
class Trainer:
	#Instantiate by pulling the information
	def __init__(self,dataobj,networkobj):
		self.data 	= dataobj
		self.net 	= networkobj
		#Only set tensors if it's a 'normal' installation
		if self.data.getProcess() != 'Other':
			self.net.setAll(self.data)
		#Set up everything here
		name = CONFIG.MAIN['Data']
		Tpath = self.data.pobj.getMPath(name)
		trainmodloc = 'Modules.' + name
		Tfile = '.Trainer_' + name
		#Check for Training Step
		if(os.path.exists(Tpath)):
   			print(name + ' Training Step Present.')
			self.Nmod = i.import_module(Tfile, package = trainmodloc)
			try:
				self.Nmod = i.import_module(Tfile, package = trainmodloc)
			except ImportError:
				sys.exit('Failed to import, trainer name fail: ' + trainmodloc)
		else:
    			print(name + ' Training Step Not Present.')
    			quit()
	def TBPrint(self):
		print('CALL THE FOLLOWING COMMAND FOR TENSORBOARD: ')
       		print('tensorboard --logdir=run1:/user/jhenzerling/work/neunet/Output/Graphs/ --port 8008')	
	def RunTraining(self):
		self.TBPrint()
		self.Nmod.Training(self.data,self.net)
		
	

























