# PDIN  
This is the code used in the paper "PDIN: A Progressive image rain removal network based on camera imaging principle". The paper is under submission.  
If you want to use this code, you need to download the following library files:  
opencv-python  
pytorch  
tensorboardX  
h5py  
scikit-image == 0.15.0  
# Testing  
We provide part of the real scene image data set for the result test. If you want to test the results of Rain100L and Rain100H, please download the respective datasets in advance.  
The required datasets and pretraining model can be downloaded using the links below:  
 [pretraining_model]()  
 [dataset]()  
To test this code, you can use the following code:  
```
 python test.py --data_path (data_path) --save_path (save_path) --log_path (log_path)
```
For example, if you want to test the results of the real data set we provide, you need to enter:  
```
 python test.py --data_path datasets/real --save_path results/real --log_path logs/Rain100H/net_latest.pth
```
# Trainging  
If you want to train your own dataset, put the dataset in the datasets folder. Note: You need to change the file name in the Dataset.py file.  
To train this code, you can use the following code:  
```
 python train.py --data_path (data_path)  --log_path (log_path)
```
If you need to change other parameters, you can read train.py.  
Additional instructions will be updated shortly.
