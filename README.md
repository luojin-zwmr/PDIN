# PDIN  
This is the code used in the paper ___"PDIN: A Progressive image rain removal network based on camera imaging principle"___. The paper is under submission. This is a relatively lightweight image rain removal algorithm. We believe that the blurring outside the depth of field caused by camera shooting must be taken into account for image dewatering tasks that are used for outdoor scene images in most cases. To this end, our method models rainfall images from both the rain-stripe Angle and the camera Angle. Our approach strikes a good balance between the effectiveness of rain removal and the cost of time.
Our paper is a relatively simple idea at the moment, but we hope that our idea of thinking about images to be processed from the perspective of camera imaging will be useful for other research. Subsequent in-depth studies will also continue to follow.  
If you want to use this code, you need to download the following library files:  
opencv-python  
pytorch  
tensorboardX  
h5py  
scikit-image == 0.15.0  
We provide part of the real scene image dataset for the result test. If you want to test the results of Rain100L and Rain100H, please download the respective datasets in advance.  
The required datasets and pretraining model can be downloaded using the links below:  
 [pretraining_model]()  
 [dataset]()  
# Testing  
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
