# PDIN
This is the code used in the paper "PDIN: A Progressive image rain removal network based on camera imaging principle". The paper is under submission.  
If you want to use this code, you need to download the following library files:  
opencv-python  
pytorch  
tensorboardX  
h5py  
Scikit - image == 0.15.0  
# Testing  
We provide part of the real scene image data set for the result test. If you want to test the results of Rain100L and Rain100H, please download the respective datasets in advance.  
The required datasets and pretraining model can be downloaded using the links below:  
 [pretraining_model]()  
 [dataset]()  
To test this code, you can use the following code:  
```
 python test.py --save_path (save_dir) --log_path (log_path)
```

