The implementation for Multi-Modal Dynamic Memory Network (MM-DMN) with tensorflow.

## Descriptions of codes

### (1) ./model
        This folder contains all of the models we mentioned in the report.   
	* DMN_single_A.py: Dynamic memory network with MFCC audio features (corresponding to DMN-A)  
	* DMN_single_VM.py: Dynamic memory network with VGG visual features or C3D motion features  (corresponding to DMN-V and DMN-M)  
	* Multimodal_DMN_VA.py: Bi-Modal dynamic memory network with visual and audio features (corresponding to DMN-VA)  
	* Multimodal_DMN_VM.py: Bi-Modal dynamic memory network with visual and motion features (corresponding to DMN-VM)  
	* Triplemodal_DMN.py: Triple-Modal dynamic memory network with visual, motion and audio features. In this setting, the question does not guide the fusion of the multi-modal features. (corresponding to DMN-VMA)  
	* TripleAttentiveModal_DMN.py: Triple-Modal dynamic memory network with visual, motion and audio features. In this setting, the question guides the fusion of multi-modal signals with attention mechanism. (corresponding to MM-DMN)  
	
### (2) ./run_xxx.py
        Control the training, testing and validation of the model  
	* each model in the './model' folder has a corresponding run_xxx.py  
	* run_dmn_single_audio.py => DMN_single_A.py  
	* run_dmn_single_vOm.py => DMN_single_VM.py  
	* run_multimodal_dmn_va.py => Multimodal_DMN_VA.py  
	* run_multimodal_dmn_vm.py => Multimodal_DMN_VM.py  
	* run_triplemodal_dmn.py => Triplemodal_DMN.py  
	* run_triple_attentive_modal_dmn.py => TripleAttentiveModal_DMN.py  
	
### (3) ./config.py
        Control the hyper-parameters and data path of the model.  

### (4) ./preprocess_msrvttqa.py & preprocess_msvdqa.py
        Preprocess the dataset MSVD and MSRVTT.  
	* Note that results on the MSVD dataset are not provided in my report. But you can still obtain the results by running the code.  
	* Also note that, videos in MSVD dataset do not have audio signals. Therefore, all the DMN models with audio features are disabled.  
	
### (5) ./util: This folder contains the codes for data_provider, basic feature extraction network, and evaluation metrics.  


## Run the code

### (1) Take the model TripleAttentiveModal_DMN as an example:  
	* python run_triple_attentive_modal_dmn.py --dataset msrvtt_qa --gpu 0 --config 0 --log msrvtt_qa_TripleAttentiveModel --mode train  
	* 'dataset': msrvtt_qa or msvd_qa  
	* 'gpu': gpu id, depends on your server  
	* 'config': config id (In my implementation, there is only one parameter configuration, and therefore the config id is 0)  
	* 'log': Log folder name. A log dir will be created with this name. The dir will contain three subdirs: stats(train, test and validation results), checkpoint(tensorflow saved model), and summary(tensorboard).  
	* 'model': train or test  
 
