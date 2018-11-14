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
from Modules.SEMSEGTUT import config_SEMSEGTUT as S
from Modules.SEMSEGTUT import Loader_SEMSEGTUT as L
from NEUnet import config_NEUnet as CM


#Tensorflow Libraries
import tensorflow as tf

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

image2d, label2d = get_entry(ENTRY,tepath)
input_shape  = [1,image2d.size]
image_data = np.array(image2d).reshape(input_shape)
image_dump_steps = np.concatenate((np.arange(0,100,20), 
                                   np.arange(100,400,100), 
                                   np.arange(400,1000,200), 
                                   np.arange(1000,20000,500))).astype(np.int32)

def Training(dobj,nobj):
	t = dobj.getResult()
	while t.current_iteration() < t.iterations():
	    t.train_step()
	    if t.current_iteration() in image_dump_steps:
		print('Image dump @ iteration {:d}'.format(t.current_iteration()))
		
		softmax, = t.ana(input_data = image_data)
		fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(24,8), facecolor='w')
		# save images
		ax0.imshow(image2d, interpolation='none', cmap='jet', vmin=0, vmax=1000, origin='lower')
		ax0.set_title('image',fontsize=24)
		
		ax1.imshow(softmax[0,:,:,0], interpolation='none', cmap='jet', vmin=0, vmax=1.0, origin='lower')
		ax1.set_title('background score',fontsize=24)
		
		ax2.imshow(softmax[0].argmax(axis=2), interpolation='none', cmap='jet', vmin=0., vmax=3.1, origin='lower')
		ax2.set_title('classification', fontsize=24)
		#plt.savefig('/user/jhenzerling/work/NEUsoft/Output/Images/iteration_{:04d}.png'.format(t.current_iteration()))




















