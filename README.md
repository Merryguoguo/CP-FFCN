# CP-FFCN

This code is for the paper Blind single-image-based thin cloud removal using a cloud perception integrated fast Fourier convolutional network ([paper](https://doi.org/10.1016/j.isprsjprs.2023.10.014)).  
****

## Introduction

This model does two things  
  1. cloud percation  
  2. cloud removal  
Targets:  
  - Simulated thin clouds  
  - Natural thin clouds  
  - Natural small-scale thick clouds  
Points:
  + Fourier attention
  + Fourier convolution
  + Blind Cloud Removal
  

****
## Preliminary

  1. Environment  
     Find it in env.txt  
  2. Training data preparation  
     Specifically, two points should be under consideration  
     - Train with simulated data created by the algorithm introduced in the paper
     - Adoption of the training strategy proposed in the paper  
  3. Commands for Training and Testing are included in ./CP-FFCN.txt  
****  

