# Harmonious Feature Learning for Interactive Hand-Object Pose Estimation 

## Directory

### Data  
You need to follow directory structure of the `data` as below.  
```  
${ROOT}  
|-- data  
|   |-- HO3D
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |   |-- evaluation
|   |   |   |-- ho3d_train_data.json
|   |   |   |-- train_segLabel 
|   |-- DEX_YCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ......
|   |   |   |--object_render
|   |   |   |--dex_ycb_s0_train_data.json
|   |   |   |--dex_ycb_s0_test_data.json
``` 

  
### Pytorch MANO layer
* For the MANO layer, I used [manopth](https://github.com/hassony2/manopth). The repo is already included in `manopth`.
* Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from [here](https://mano.is.tue.mpg.de/) and place at `assets/mano_models`.

## Acknowledgments
We thank: 
* [Semi-Hand-Object](https://github.com/stevenlsw/Semi-Hand-Object.git) 
* [HandOccNet](https://github.com/namepllet/HandOccNet.git)


