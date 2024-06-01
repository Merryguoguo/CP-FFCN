# CP-FFCN

This code is for the [paper](https://doi.org/10.1016/j.isprsjprs.2023.10.014):   
Blind single-image-based thin cloud removal using a cloud perception integrated fast Fourier convolutional network.  
****

## Introduction :bulb:

:sunny: The model does two things:  
  1. Cloud percation  
  2. Cloud removal

:star: Targets:  
  - Simulated thin clouds  
  - Natural thin clouds  
  - Natural small-scale thick clouds

:dizzy: Points:
  + Fourier attention -- Cloud perception  
  + Fourier convolution -- Cloud removal  
  

****
## Preliminary :anchor:

  1. Environment :earth_africa:  
     Find it in [env.txt](https://github.com/Merryguoguo/CP-FFCN/blob/main/env.txt)  
  2. Training data preparation :pushpin:  
     Take two points under consideration:  
     - Training with simulated data created by the [algorithm](https://github.com/Merryguoguo/Simu-clouds.git) introduced in the paper
     - Adoption of the training strategy proposed in the [paper](https://doi.org/10.1016/j.isprsjprs.2023.10.014).  
  3. Commands for Training and Testing are included in ./CP-FFCN.txt :spiral_notepad: 
**** 
## Pretrained :film_projector:
For models trained with the [RICE dataset](https://paperswithcode.com/dataset/rice) and [STGAN dataset](https://openaccess.thecvf.com/content_WACV_2020/papers/Sarukkai_Cloud_Removal_from_Satellite_Images_using_Spatiotemporal_Generator_Networks_WACV_2020_paper.pdf), please find them [here](https://drive.google.com/file/d/12IXY2asM2aREp9BMJNzHR-qnpdR5Nkbz/view?usp=drive_link).  
The models have been tesified on the tasks of simulated cloud removal, real-world thin cloud removal, and small-scale thick cloud removal.  
More specific scenarios for model testing please find in the [paper](https://doi.org/10.1016/j.isprsjprs.2023.10.014).
**** 
## Representation :film_strip:
#### :cloud: Natural Thin Cloud Removal:  
![image](https://github.com/Merryguoguo/CP-FFCN/assets/54757576/a9d4b57b-c02c-4fab-a720-97cc669a8b70)  

#### :cloud: Natural Small-scale Thick Cloud Removal:  
![image](https://github.com/Merryguoguo/CP-FFCN/assets/54757576/6b57f15f-1520-4f5e-898e-e28c6f5b978f)  
  
![image](https://github.com/Merryguoguo/CP-FFCN/assets/54757576/cc24cc7c-8579-431d-8b5e-75aa49148067)

## Citing CP-FFCN
If you find this code/data useful in your research then please cite our [paper](https://doi.org/10.1016/j.isprsjprs.2023.10.014):
```
Guo, Y., He, W., Xia, Y., & Zhang, H. (2023). Blind single-image-based thin cloud removal using a cloud perception integrated fast Fourier convolutional network. ISPRS Journal of Photogrammetry and Remote Sensing, 206, 63–86. https://doi.org/10.1016/j.isprsjprs.2023.10.014
```
