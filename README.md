# CP-FFCN

This code is for the paper Blind single-image-based thin cloud removal using a cloud perception integrated fast Fourier convolutional network ([paper](https://doi.org/10.1016/j.isprsjprs.2023.10.014)).  
****

## Introduction

:sunny: This model does two things:  
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
## Preliminary

  1. Environment  
     Find it in env.txt  
  2. Training data preparation  
     Take two points under consideration:  
     - Training with simulated data created by the algorithm introduced in the paper
     - Adoption of the training strategy proposed in the paper  
  3. Commands for Training and Testing are included in ./CP-FFCN.txt  
****  

## Representation
:cloud: Natural Thin cloud removal:  
![image](https://github.com/Merryguoguo/CP-FFCN/assets/54757576/a9d4b57b-c02c-4fab-a720-97cc669a8b70)  

:cloud: Natural small-scale Thick cloud removal:  
![image](https://github.com/Merryguoguo/CP-FFCN/assets/54757576/6b57f15f-1520-4f5e-898e-e28c6f5b978f)  
![image](https://github.com/Merryguoguo/CP-FFCN/assets/54757576/cc24cc7c-8579-431d-8b5e-75aa49148067)

