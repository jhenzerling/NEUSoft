#Finder class for file-handling and extra, gets paths and passes them around
import os,sys,os.path
import importlib as i
from NEUnet import config_NEUnet as CONFIG

class Finder:
	#Instantiate Paths
	def __init__(self):
		self.PATH = CONFIG.PATH['Path']
		self.MPATH = self.PATH + '/Modules'
		self.NPATH = self.PATH + '/Networks'	
	#Modules Methods
	def setMPath(self,inputs):
		self.MPATH = inputs
	def getMPath(self,name):
		return self.MPATH
	#Data Methods
	def setDPath(self,inputs):
		self.DPATH = inputs
	def getDPath(self,name):
		self.DPATH = self.MPATH + '/' + name + '/Data'
		if(os.path.exists(self.DPATH)):
   			return self.DPATH
		else:
    			print(name + ' Data Path Fail')
    			quit()
	#Network Architecture Methods
	def setAPath(self,inputs):
		self.APATH = inputs
	def getAPath(self,name):
		self.APATH = 'Networks' + '/' + name
		if(os.path.exists(self.APATH)):
   			return self.APATH
		else:
    			print(name + ' Architecture Path Fail')
    			quit()
	

    
