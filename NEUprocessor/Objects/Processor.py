#Imager class for handling image display, will add Imager_DATA to Modules
import os,sys,os.path
import importlib as i
from NEUprocessor import config_NEUprocessor as CONFIG

class Processor:
	#Instantiate Paths
	def __init__(self):
		Mpath = CONFIG.PATH['Path']
		Ipath = Mpath + '/Modules/' + CONFIG.MAIN['Data'] + '/'
		procname = 'Processor_' + CONFIG.MAIN['Data']
		procmodloc = 'Modules.' + CONFIG.MAIN['Data']
		Ifile = Ipath + procname + '.py' 
		
		if(os.path.exists(Ipath)):
   			print(CONFIG.MAIN['Data'] + ' Module Present.')
			#Check for Imager
			if(os.path.isfile(Ifile)):
   				print(CONFIG.MAIN['Data'] + ' Processor Present.')
				try:
					self.Imod = i.import_module('.' + procname, package = procmodloc)
				except ImportError:
					sys.exit('Failed to import, processor name fail: ' + procmodloc)	
			else:
    				print(CONFIG.MAIN['Data'] + ' Processor Not Present.')    		
				quit()
		else:
    			print(CONFIG.MAIN['Data'] + ' Module Not Present.')
    			quit()

	###################################

	def preProcess(self,path,options):
		self.Imod.Process(path,options)
	
