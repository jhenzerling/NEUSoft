#Loader Object to Pull from Loader Folders
import os,sys,os.path
import importlib as i
import tensorflow as tf
class Loader:
	#######################################################################################
	#######################################################################################
	#Initializer by checking if Data is Present then Importing
	def __init__(self,dname,pather):
		#Finder Object called via Data Object
		self.pobj = pather
		#Set Names/Paths
		name = dname
		Dpath = self.pobj.getDPath(name)
		Mpath = self.pobj.getMPath(name)
		loadername = 'Loader_' + name
		Lfile = Mpath + '/' + name + '/' + loadername + '.py'
		loadermodloc = 'Modules.' + name
		self.Lmod = i.import_module('.' + loadername, package = loadermodloc)
		#Check for Data
		if(os.path.exists(Dpath)):
   			print(name + ' Data Present.')
			#Check for Loader
			if(os.path.isfile(Lfile)):
   				print(name + ' Loader Present.')
				try:
					self.Lmod = i.import_module('.' + loadername, package = loadermodloc)
				except ImportError:
					sys.exit('Failed to import, loader name fail: ' + loadermodloc)	
			else:
    				print(name + ' Loader Not Present.')    		
				quit()
		else:
    			print(name + ' Data Not Present.')
    			quit()
	#######################################################################################
	#######################################################################################
	#Getter for Image Information
	def getImageInfo(self):
		return [self.Lmod.hsize,self.Lmod.vsize,self.Lmod.colours,self.Lmod.cnumb]
	#Getter for Type Information
	def getVariety(self):
		return self.Lmod.variety
	def getProcess(self):
		return self.Lmod.process
	#Getter for Set Information
	def getSetInfo(self):
		return [self.Lmod.trainsize,self.Lmod.testsize]
	#Getter for Result of Loader
	def getResult(self):
		return self.Lmod.result
	#Getter for CFG's
	def getCFG(self):
		return self.Lmod.dataCFG
	#Getter for Mod
	def getMod(self):
		return self.Lmod
	#Form TF placeholders
	def getPlaceholders(self):	
		FlatT 	= tf.placeholder(tf.float32,
			[None,self.Lmod.hsize*self.Lmod.vsize*self.Lmod.colours], name='flat')
		LabelT 	= tf.placeholder(tf.float32,[None,self.Lmod.cnumb], name='label')
		TwoDT	= tf.reshape(FlatT,
			[-1,self.Lmod.hsize,self.Lmod.vsize,self.Lmod.colours], name='2D')
		return [FlatT,TwoDT,LabelT]
	#######################################################################################
	#######################################################################################


    
