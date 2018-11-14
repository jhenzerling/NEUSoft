#PSingle HANDLING
#from __future__ import print_function

import numpy as np
import tensorflow as tf

import ROOT as R
import time

from larcv import larcv
from larcv.dataloader2 import larcv_threadio

from NEUnet import config_NEUnet as CM
from Modules.PSINGLE import config_PSINGLE as C

larcv.load_pyutil()

#Type of problem being solved and way data is stored and name
variety 	= C.DATA['Variety']
process		= C.DATA['Process']
name		= CM.MAIN['Data']
#Image Dimensions and Colours and Classes and Axes
hsize 		= C.DATA['H']
vsize 		= C.DATA['V']
colours 	= C.DATA['Colours']
cnumb 		= C.DATA['Classes']
#Batchsize designates number of points in each separate chunk
#Stepnumber designates number batches to go over
trbatchsize 	= CM.TRAINING['Training Batch Size']
tebatchsize	= CM.TRAINING['Testing Batch Size']
#Files and Paths
trf 		= C.DATA['Training File']
tef 		= C.DATA['Testing File']
path 		= '/user/jhenzerling/work/NEUsoft/Modules/PSINGLE/Data/'
trpath 		= path + trf
tepath 		= path + tef
#Collect the Config File
Train_cfg 	= C.DATA['Training CFG']
Test_cfg 	= C.DATA['Testing CFG']   
     
##################################################################
#PSingle Methods

#Construct and prepare memory for the threadio
def IOPrep(name,b):
    if(name == 'train'):
        cfg = Train_cfg
    elif(name == 'test'):
        cfg = Test_cfg
    else:
        print('Bad name, check CFG')
    
    proc = larcv_threadio()
    proc.configure(cfg)
    proc.start_manager(b)
    #Need sleep for manager to finish loading
    time.sleep(2)
    proc.next()

    return proc

#Convert the flat data into a 2D image for use in the network
def formatTensors(da,bs):
    image_tensor = tf.convert_to_tensor(da[0])
    image_tensor_2d = tf.reshape(image_tensor, [bs, hsize, vsize, colours])
    label_tensor = tf.convert_to_tensor(da[1])
    
    return [image_tensor_2d, label_tensor]

#########################################################################################
#Gives range of energies and amount of each pdg from get-go, this is an example for further data
def spread():
	#0elec,1gamm,2muo,3pio,4prot
	PDG2NAME = {11   : 0,
                    22   : 1,
                    13   : 2,
                    211  : 3,
                    2212 : 4}	
	bnumb = 20
	chain = R.TChain("particle_mctruth_tree")
	chain.AddFile(trpath)
	pdgbin = np.zeros(cnumb)
	ebin = np.zeros(chain.GetEntries())
	#Store pdg's and init energies
	for x in range(chain.GetEntries()):
		chain.GetEntry(x)
		truth = chain.particle_mctruth_branch
		parray = truth.as_vector()
		pdgbin[PDG2NAME[parray[0].pdg_code()]] += 1
		ebin[x] = parray[0].energy_init()*1000 - larcv.ParticleMass(parray[0].pdg_code())
	#Sort the array and find the range of TOTAL energies (kin+mass)
	ebin2 = np.sort(ebin)
	low = ebin2[0]
	high = ebin2[-1]
	dist = (high - low)/bnumb
	ebin3 = np.zeros(bnumb)
	for x in range(len(ebin)):
		for y in range(bnumb):
			if (low + y*dist <= ebin2[x] <= low + (y+1)*dist):
			  	ebin3[y] += 1
	return[pdgbin,ebin3]

#########################################################################################
#Data Handling
trproc = IOPrep('train',trbatchsize)
teproc = IOPrep('test',tebatchsize)

result = [trproc,teproc]
dataCFG = [Train_cfg,Test_cfg]










