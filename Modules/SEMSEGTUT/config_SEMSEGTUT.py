#Configuration for PSINGLE Data
#Configuration for Data
DATA= {
    	'H'       		: 256,
    	'V'       		: 256,
	'V2'			: 1666,
    	'Colours' 		: 1,
	'Axes'			: 3,
	'Classes' 		: 3,
    	'Training Set Size'	: 15000,
	'Testing Set Size'	: 10000,
	'Data Path'    		: '/user/jhenzerling/work/NEUsoft/Modules/SEMSEGTUT/Data/',
	'Training File'     	: 'train_15k.root',
   	'Training CFG'   	: {'filler_name':'TrainCFGSEG','verbosity':0,'filler_cfg':'Modules/SEMSEGTUT/TrainCFGSEMSEGTUT.cfg'},
    	'Testing File'     	: 'test_10k.root',
    	'Testing CFG'   	: {'filler_name':'TestCFGSEG','verbosity':0,'filler_cfg':'Modules/SEMSEGTUT/TestCFGSEMSEGTUT.cfg'},
	'Variety'		: 'Segmentation',
	'Process'		: 'Other'	
}

