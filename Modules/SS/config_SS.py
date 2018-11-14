#Configuration for PSINGLE Data
#Configuration for Data
DATA= {
    	'H'       		: 256,
    	'V'       		: 256,
	'V2'			: 1666,
    	'Colours' 		: 1,
	'Axes'			: 3,
	'Classes' 		: 3,
    	'Training Set Size'	: 9000,
	'Testing Set Size'	: 8500,
	'Data Path'    		: '/user/jhenzerling/work/NEUsoft/Modules/SS/Data/',
	'Training File'     	: 'SSTrain.root',
   	'Training CFG'   	: {'filler_name':'TrainCFGSEG','verbosity':0,'filler_cfg':'Modules/SS/Train_CFG_SS.cfg'},
    	'Testing File'     	: 'SSTest.root',
    	'Testing CFG'   	: {'filler_name':'TestCFGSEG','verbosity':0,'filler_cfg':'Modules/SS/Test_CFG_SS.cfg'},
	'Variety'		: 'Segmentation',
	'Process'		: 'Other'	
}

