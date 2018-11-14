#This module groups up the learning techniques so it's easier
#Numpy for number-handling
import numpy as np

#File Handling
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys,os
import ROOT as R
DB = R.TDatabasePDG()

from larcv import larcv
from larcv.dataloader2 import larcv_threadio
larcv.load_pyutil()
#Suppress extra warnings for readability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Machine Learning Handling
from Modules.SS import config_SS as S
from Modules.SS import Loader_SS as L
from NEUnet import config_NEUnet as CM


#Tensorflow Libraries
import tensorflow as tf

#OH!!! IT DOESNT USE THE DAG NABBED THREADIO?!!?
ENTRY=2
def get_entry(entry,tepath):
    # image
    chain_image2d = R.TChain("image2d_data_tree")
    chain_image2d.AddFile(tepath)
    chain_image2d.GetEntry(entry)
    cpp_image2d = chain_image2d.image2d_data_branch.as_vector().front()
    # label
    chain_label2d = R.TChain("image2d_segment_tree")
    chain_label2d.AddFile(tepath)
    chain_label2d.GetEntry(entry)
    cpp_label2d = chain_label2d.image2d_segment_branch.as_vector().front()    
    return (np.array(larcv.as_ndarray(cpp_image2d)), np.array(larcv.as_ndarray(cpp_label2d)))


tef 		= S.DATA['Testing File']
path 		= S.DATA['Data Path']
tepath 		= path + tef

#image2d, label2d = get_entry(ENTRY,tepath)
#input_shape  = [1,image2d.size]
#image_data = np.array(image2d).reshape(input_shape)
#image_dump_steps = np.concatenate((np.arange(0,100,20), 
#                                   np.arange(100,400,100), 
#                                   np.arange(400,1000,200), 
#                                   np.arange(1000,20000,500))).astype(np.int32)

def Training(dobj,nobj):
	t = dobj.getResult()
	RunImage(t,1,0)
	#softie = t.ana(input_data = data)
	#output = softie[0].argmax(axis=2)
	#plt.imshow(output,cmap=plt.get_cmap())
	while t.current_iteration() < t.iterations():
	    t.train_step()
	

def RunImage(t,which,entry):
	deg = 3
	counter = np.zeros(deg)
	d1 = R.TChain('image2d_sbndwire_tree')
	d2 = R.TChain('image2d_sbnd_cosmicseg_tree')
	if which == 1:
		dname3 = 'SSTrain2.root'
		amount = 9000
	elif which == 2:
		dname3 = 'SSTest2.root'
		amount = 8500
	else:
		print('eff')
	print(dname3)
	fpath3 = '/user/jhenzerling/work/NEUsoft/Modules/SS/Data/' + dname3
	d1.AddFile(fpath3)
	d2.AddFile(fpath3)
	d1.GetEntry(entry)
	d2.GetEntry(entry)
	d1b = d1.image2d_sbndwire_branch
	d2b = d2.image2d_sbnd_cosmicseg_branch
	d1v = d1b.as_vector()
	d2v = d2b.as_vector()
	d1i = larcv.as_ndarray(d1v.front())
	d2i = larcv.as_ndarray(d2v.front())
	d1i2 = d2i.reshape((1,655360))
	
	softmax = t.ana(input_data = d1i2)
	#print(softmax[0].argmax(axis=0).shape)
	#print(softmax[0].argmax(axis=1).shape)
	#print(softmax[0].argmax(axis=2).shape)
	#print(softmax[0].argmax(axis=3)[0].shape)
	#print(d2i.shape)
	output = softmax[0].argmax(axis=3)[0]
	fig, (ax0,ax1) = plt.subplots(1,2,figsize=(24,8), facecolor='w')
	ax0.imshow(output,cmap=plt.get_cmap())
	ax0.set_title('run-net',fontsize=24)
	ax1.imshow(d2i,cmap=plt.get_cmap())
	ax1.set_title('raw',fontsize=24)
	plt.show()


















