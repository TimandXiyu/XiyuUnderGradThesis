# XiyuUnderGradThesis
---
## Intro: 
+ This is the repo for Xiyu Wang doing his undergrad thesis experiments.  
+ All of those codes are based on PyTorch.  
+ Current experiments that had been done are Unet, DeepLabv3+ and D-LinkNet with DeepGlobe dataset.  
---
## To do list:  
1. Baseline Benchmarking (on-going)  
2. Generate Fake Dataset (next)  
3. Benchmarking Again  

### 2020/12/29 update:  
+ Validate following dataset on Deepglobe itself.
+ Dice and IoU can be interchanged, thus are actually same stuff in different mathematical version  
+ Dice Loss is generally better than IoU loss due to dice loss's numerical values are greater thus gradients are larger when backproping  

| network     | dice  | res     | loss     | bs | optimizer | augmentation       |
|:-----------:|:-----:|:-------:|:--------:|:--:| :-------: | :----------------: |
| Unet        | 0.63  | 1024^2  | BCE+dice | 2  | RMSprop   | None               |
| DeepLabv3+  | 0.43  | 1024^2  | BCE+dice | 2  | Adam      | V+H Flip, Gaus, HSV|
| D-LinkNet34 | 0.778 | 1024^2  | BCE+dice | 4  | Adam      | V+H Flip, Gaus, HSV|
| Unet        | 0.67  | 1024^2  | BCE+dice | 2  | Adam      | V+H Flip, Gaus, HSV|

### 2020/12/30 to do this week:  
+ ~~Finish the cropping of Chongzhou and Wuzhen image~~
+ Looking into style transfer network  
+ Starting Igarss paper  

#### Extra Comments  
The cropping tool can crop and rotate at the same time  
![Image text](https://github.com/TimandXiyu/XiyuUnderGradThesis/blob/main/readme_img/readme_image_1.png)  
If you wang to solely use the cropping tool, it is in the utils directory.  
If you don't know how to use the cropping tool, please open a issue!  
