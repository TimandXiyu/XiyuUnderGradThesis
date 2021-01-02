# XiyuUnderGradThesis
---
## Intro: 
+ This is the repo for Xiyu Wang doing his undergrad thesis experiments.  
+ All of those codes are based on PyTorch.  
+ Current experiments that had been done are Unet, DeepLabv3+ and D-LinkNet with DeepGlobe dataset.  
---
## General to do list:  
1. ~~Baseline Benchmarking~~  
2. ~~Generate Fake Dataset~~  
3. Benchmarking Again (on going)  

### Weekly to do:  
+ ~~Finish the cropping of Chongzhou and Wuzhen image
+ ~~Looking into style transfer network  
+ Starting Igarss paper  


##Main Results (updated at 2020/1/3)  
| network     | dice  | res     | loss     | bs | optimizer | augmentation       | Cross-area teston cz | Cross-area test on wz | Cross-area testing with extra 1000 fake img |
|:-----------:|:-----:|:-------:|:--------:|:--:| :-------: | :----------------: | :------------------: | :-------------------: | :-----------------------------------------: |
| Unet        | 0.63  | 1024^2  | BCE+dice | 2  | RMSprop   | None               | None                 | None                  | None |
| DeepLabv3+  | 0.43  | 1024^2  | BCE+dice | 2  | Adam      | V+H Flip, Gaus, HSV| None                 | None                  | None |
| D-LinkNet34 | 0.778 | 1024^2  | BCE+dice | 4  | Adam      | V+H Flip, Gaus, HSV| 0.62                 | None                  | 0.67 |
| Unet        | 0.67  | 1024^2  | BCE+dice | 2  | Adam      | V+H Flip, Gaus, HSV| 0.61                 | 0.62                  | to do |
| D-LinkNet101| 0.78  | 1024^2  | BCE+dice | 4  | Adam      | V+H Flip, Gaus, HSV| 0.63                 | 0.64                  | to do |

### 2020/12/29 update:  
+ Validate following dataset on Deepglobe itself  
+ Dice and IoU can be interchanged, thus are actually same stuff in different mathematical version  
+ Dice Loss is generally better than IoU loss due to dice loss's numerical values are greater thus gradients are larger when backproping  

### 2021/01/03 update:
+ Style transfering with default pre-trained model, not very satisfying results. Usuing pure random pair  
+ Generated 1000 images for training  
+ Fine Tune was clearly outperformed by training from scratch with mixed data  
+ Notice that dice=0.67 actually means IoU=0.5 (Calculate from (2-Dice)/Dice)

# Cropping tool usage
The cropping tool can crop and rotate at the same time  
![Image text](https://github.com/TimandXiyu/XiyuUnderGradThesis/blob/main/readme_img/readme_image_1.png)  
If you wang to solely use the cropping tool, it is in the utils directory.  
If you don't know how to use the cropping tool, please open an issue!  


