# Harmonious Feature Learning for Interactive Hand-Object Pose Estimation 

## Directory
```  
${ROOT}  
|-- data  
|   |-- HO3D
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |   |-- evaluation
|   |   |   |-- train_segLable
|   |   |   |-- ho3d_train_data.json
|   |-- DEX_YCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ......
|   |   |   |-- object_render
|   |   |   |-- dex_ycb_s0_train_data.json
|   |   |   |-- dex_ycb_s0_test_data.json
```

### Data  
You need to follow directory structure of the `data` as below.  

* Download HO3D(version 2) data [data](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/)
* Download DexYCB data [data](https://dex-ycb.github.io/)
* Download the process data [data](https://drive.google.com/drive/folders/1QnggoyWgZLuewWBDh4dTDuvr0UlJILNv?usp=drive_link)

  
### Pytorch MANO layer
* For the MANO layer, I used [manopth](https://github.com/hassony2/manopth). The repo is already included in `manopth`.
* Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from [here](https://mano.is.tue.mpg.de/) and place at `assets/mano_models`.

### Train  
#### HO3d
```
sh sh/train_ho3d.sh
```
#### Dex-ycb
```
sh sh/train_dex-ycb.sh
```
### Test  
#### HO3d
```
sh sh/train_ho3d_test.sh
```
#### Dex-ycb
```
sh sh/train_dex-ycb_test.sh
```  

## Acknowledgments
We thank: 
* [Semi-Hand-Object](https://github.com/stevenlsw/Semi-Hand-Object.git) 
* [HandOccNet](https://github.com/namepllet/HandOccNet.git)


