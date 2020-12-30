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

| network     | dice  | res     | loss     | bs | optimizer |
|:-----------:|:-----:|:-------:|:--------:|:--:| :-------: |
| Unet        | 0.63  | 1024^2  | BCE+dice | 2  | RMSprop   |
| DeepLabv3+  | 0.43  | 1024^2  | BCE+dice | 2  | Adam      |
| D-LinkNet34 | 0.778 | 1024^2  | BCE+dice | 4  | Adam      |

### 2020/12/30 to do:  
+ Finish the cropping of Chongzhou and Wuzhen image  
+ Looking into style transfer network  
+ Starting Igarss paper  
