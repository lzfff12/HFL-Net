#from unittest.case import doModuleCleanups
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from dataset import dex_ycb_util
from dataset import dataset_util
import json
import os
#from utils.manolayer_ho3d import ManoLayer
from torchvision.transforms import functional
from manopth.manopth.manolayer import ManoLayer
import copy
mano_layer = ManoLayer(flat_hand_mean=False,
                           side="right", mano_root="assets/mano_models", use_pca=False)
mano_layerl = ManoLayer(flat_hand_mean=False,
                           side="left", mano_root="assets/mano_models", use_pca=False)

class dex_ycb(data.Dataset):
    def __init__(self,dataset_root,mode,inp_res=256 ,max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1,
                 hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5) -> None:
        #super(dex_ycb,self).__init__()
        self.root = dataset_root
        self.mode = mode
        self.inp_res = inp_res
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                      1, 2, 3, 17,
                                      4, 5, 6, 18,
                                      10, 11, 12, 19,
                                      7, 8, 9, 20]               #注意
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        #object information
        self.obj_mesh = dex_ycb_util.load_objects_dex_ycb(dataset_root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot
            with open(os.path.join(dataset_root,"dex_ycb_s0_train_data.json"), 'r', encoding='utf-8') as f:
                self.sample_dict = json.load(f)
            
            self.sample_list = sorted(self.sample_dict.keys(),key=lambda x:int(x[3:]))
            self.sample_list_processed = []
            #筛选bbox expansion_factor=1.5后仍然在图片的样本.与22年的数据处理对齐
            for sample in self.sample_list:
                joint_2d = np.array(self.sample_dict[sample]["joint_2d"],dtype=np.float32).squeeze()
                hand_bbox = dex_ycb_util.get_bbox(joint_2d, np.ones_like(joint_2d[:,0]), expansion_factor=1.5)
                hand_bbox = dex_ycb_util.process_bbox(hand_bbox,640,480,expansion_factor=1.0)
                if hand_bbox is None:
                    continue
                else:
                    self.sample_list_processed.append(sample)

        else:
            self.hand_bbox_list = []
            with open(os.path.join(dataset_root,"dex_ycb_s0_test_data.json"), 'r', encoding='utf-8') as f:
                self.sample_dict = json.load(f)
            self.sample_list = sorted(self.sample_dict.keys(),key=lambda x:int(x[3:]))
            #筛选bbox expansion_factor=1.5后仍然在图片的样本.与22年的数据处理对齐
            idx = 0
            for sample in self.sample_list:
                #if idx%100==0:
                #    print(self.sample_dict[sample]["color_file"])
                idx = idx + 1
                joint_2d = np.array(self.sample_dict[sample]["joint_2d"],dtype=np.float32).squeeze()
                hand_bbox = dex_ycb_util.get_bbox(joint_2d, np.ones_like(joint_2d[:,0]), expansion_factor=1.5)
                hand_bbox = dex_ycb_util.process_bbox(hand_bbox,640,480,expansion_factor=1.0)
                if hand_bbox is None:
                    hand_bbox = np.array([0,0,640-1, 480-1], dtype=np.float32)
                self.hand_bbox_list.append(hand_bbox)
            self.sample_list_processed = self.sample_list 
    
    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)

        # Randomly jitter center
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        scale = scale * scale_jittering

        #rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        rot_factor = 30
        rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
        rot = rot*self.max_rot/180
        
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(center, scale,
                                                                              [self.inp_res, self.inp_res], rot=rot,
                                                                              K=K)
        # Change mano from openGL coordinates to normal coordinates
        # 注意！
        mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=np.eye(3))

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = dataset_util.color_jitter(img, brightness=self.brightness,
                                        saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        #print(obj_mask)
        obj_mask = torch.from_numpy(obj_mask)

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj
    
    def data_crop(self, img, K, hand_joints_2d, p2d):
        crop_hand = dataset_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(hand_joints_2d, bbox_factor=1.1)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(center, scale, [self.inp_res, self.inp_res])
        bbox_hand = dataset_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        return img, affinetrans, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.sample_list_processed)

    def __getitem__(self,idx):
        sample = {}
        sample_info = self.sample_dict[self.sample_list_processed[idx]]
        do_flip = (sample_info["mano_side"] == 'left') 
        img =Image.open(os.path.join(self.root,sample_info["color_file"])).convert("RGB")
        #camintr
        fx = sample_info['intrinsics']['fx']
        fy = sample_info['intrinsics']['fy']
        cx = sample_info['intrinsics']['ppx']
        cy = sample_info['intrinsics']['ppy']
        K = np.zeros((3,3))
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
        K[2,2] = 1
        if do_flip:
            img = np.array(img, np.uint8 , copy=True)
            img = img[:, ::-1, :]
            img = Image.fromarray(np.uint8(img))
        if self.mode == "train":
            #hand information
            mano_pose_pca_mean =  np.array(sample_info["pose_m"],dtype=np.float32).squeeze()
            mano_betas = np.array(sample_info["mano_betas"],dtype=np.float32)
            hand_joint_3d = np.array(sample_info["joint_3d"],dtype=np.float32).squeeze()
            joints_uv =  np.array(sample_info["joint_2d"],dtype=np.float32).squeeze()
            # #前三维是全局旋转，后45维是pose ,后三维是平移
            mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layer.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
            if do_flip:
                #注意左手的PCA转换系数与右手不同
                mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layerl.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
                #包括全局旋转在内mano系数的后两维*-1
                mano_pose_aa_mean_wo_trans = mano_pose_aa_mean[:48].reshape(-1,3)
                mano_pose_aa_mean_wo_trans[:,1:]  *= -1
                mano_pose_aa_mean[0:48] = mano_pose_aa_mean_wo_trans.reshape(-1)               
                #3D点关于Y-O-X对称
                hand_joint_3d[:,0] *= -1 
                #2D点关于W/2对称
                joints_uv[:,0] = np.array(img.size[0],dtype=np.float32)  - joints_uv[:,0] - 1
            #前三维是全局旋转，后45维是pose
            mano_pose_aa_flat = np.concatenate((mano_pose_aa_mean[:3],mano_pose_aa_mean[3:48]+mano_layer.smpl_data["hands_mean"]),axis=0)
            # #前三维是全局旋转，后45维是pose ,后十维度是shape
            mano_param = np.concatenate((mano_pose_aa_flat,mano_betas)) 

            gray = Image.open(os.path.join(self.root,sample_info["object_seg_file"]))
            if do_flip:
                gray = np.array(gray, np.uint8 , copy=True)
                gray = gray[:, ::-1]
                gray = Image.fromarray(np.uint8(gray))

            #object_information
            grasp_object_pose = np.array(sample_info["pose_y"][sample_info['ycb_grasp_ind']],dtype=np.float32)
            #grasp_object_pose[1] *= -1   #注意！
            #grasp_object_pose[2] *= -1
            p3d,p2d= dex_ycb_util.projectPoints(self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]],K,rt=grasp_object_pose)
            if do_flip:
                p2d[:,0] = np.array(img.size[0],dtype=np.float32)  - p2d[:,0] - 1
                p3d[:,0] *= -1
            
            #data augumentation
            img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj = self.data_aug(img, mano_param, joints_uv, K, gray, p2d)
            
            

        
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            #sample["cam_intr"] = K
            sample["joints2d"] = joints_uv
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
        else:
            # object
            sample["obj_bbox3d"] = self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]]
            sample["obj_diameter"] = self.obj_diameters[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]]
            grasp_object_pose = dex_ycb_util.pose_from_initial_martrix(np.array(sample_info["pose_y"][sample_info['ycb_grasp_ind']],dtype=np.float32))
            _,p2d = dex_ycb_util.projectPoints(self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]],K,rt=grasp_object_pose)
            if do_flip:
                 p2d[:,0] = np.array(img.size[0],dtype=np.float32)  - p2d[:,0] - 1
            #     grasp_object_pose[0,:] *= -1
            sample["obj_pose"] = grasp_object_pose
            sample["obj_cls"] = sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]


            #hand
            joints_uv =  np.array(sample_info["joint_2d"],dtype=np.float32).squeeze()
            if do_flip:
                joints_uv[:,0] = np.array(img.size[0],dtype=np.float32)  - joints_uv[:,0] - 1
            hand_joint_3d = np.array(sample_info["joint_3d"],dtype=np.float32).squeeze()
            root_joint = copy.deepcopy(hand_joint_3d[self.joint_root_id])
            #注意同22年的手的测试保持一致，故手的根节点以及手的关节都是相机空间中的绝对位置，且不做翻转
            sample["root_joint"] = root_joint
            #22年
            sample["hand_type"] = sample_info["mano_side"]
            sample["joints_coord_cam"] = hand_joint_3d


            #crop
            img, affinetrans, bbox_hand, bbox_obj = self.data_crop(img, K, joints_uv, p2d)

            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["cam_intr"] = K
            sample["affinetrans"]  = affinetrans

        return sample









#############################################################################################
        # #3D可视化
        # sample_info = self.sample_dict[self.sample_list[5999]]
        # img =Image.open(os.path.join(self.root,sample_info["color_file"])).convert("RGB")
        # mano_pose_pca_mean =  np.array(sample_info["pose_m"],dtype=np.float32).squeeze()
        # mano_betas = np.array(sample_info["mano_betas"],dtype=np.float32)
        # hand_joint_3d = np.array(sample_info["joint_3d"],dtype=np.float32).squeeze()
        # joints_uv =  np.array(sample_info["joint_2d"],dtype=np.float32).squeeze()
        # mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layerl.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
        # mano_param = np.concatenate((mano_pose_aa_mean[0:48],mano_betas)) 

        # import matplotlib
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # matplotlib.use('agg')
        # from utils.vis_utils import plot3dVisualize
        # #matplotlib.use('TkAgg')

        # print(matplotlib.get_backend () )
        # #准备数据
        # fl = mano_layerl.th_faces.cpu().numpy().squeeze()
        # fr = mano_layer.th_faces.cpu().numpy().squeeze()
        # ml = dict()
        # mr=dict()
        
        # th_vertsl, th_jtrl = mano_layerl(th_pose_coeffs=torch.from_numpy(mano_param[None,:48]).float(),th_betas=torch.from_numpy(mano_param[None,48:]).float())
        # mano_pose_aa_mean_wo_trans = mano_pose_aa_mean[:48].reshape(-1,3)
        # mano_pose_aa_mean_wo_trans[:,1:]  *= -1

        # mano_param = np.concatenate((mano_pose_aa_mean_wo_trans.reshape(-1),mano_betas)) 

        # th_vertsr, th_jtrr = mano_layer(th_pose_coeffs=torch.from_numpy(mano_param[None,:48]).float(),th_betas=torch.from_numpy(mano_param[None,48:]).float())

        # ml["v"] = th_vertsl.squeeze()
        # ml["f"] = fl

        # mr["v"] = th_vertsr.squeeze()
        # mr["f"] = fr



        # handMesh = ml
        # handMeshr = mr
        # fig = plt.figure(figsize=(25, 20))
        # figManager = plt.get_current_fig_manager()
        # #figManager.resize(*figManager.window.maxsize())

        # ax0 = fig.add_subplot(2, 2, 1)
        # ax0.imshow(img)
        # ax0.title.set_text('RGB Image')

        # img = np.array(img, np.uint8 , copy=True)
        # img = img[:, ::-1, :]
        # img = Image.fromarray(np.uint8(img))
        # ax1 = fig.add_subplot(2, 2, 2)
        # ax1.imshow(img)
        # ax1.title.set_text('do_flip')

        # ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        # plot3dVisualize(ax3, handMesh, flip_x=False, isOpenGLCoords=False, c="r")
        # ax3.title.set_text('Hand Mesh')

        # ax4 = fig.add_subplot(2, 2, 4 , projection="3d")
        # plot3dVisualize(ax4, handMeshr, flip_x=False, isOpenGLCoords=False, c="r")
        # ax4.title.set_text('3D Annotations projected to 2D')
        
        # plt.savefig('./test.jpg')
########################################################################################################
        # #2Ds手可视化（注意修改代码，根据图片是左手还是右手（PCA comp））
        # sample_info = self.sample_dict[self.sample_list[5999]]
        # fx = sample_info['intrinsics']['fx']
        # fy = sample_info['intrinsics']['fy']
        # cx = sample_info['intrinsics']['ppx']
        # cy = sample_info['intrinsics']['ppy']
        # #camintr
        # K = np.zeros((3,3))
        # K[0,0] = fx
        # K[1,1] = fy
        # K[0,2] = cx
        # K[1,2] = cy
        # K[2,2] = 1
        # img =Image.open(os.path.join(self.root,sample_info["color_file"])).convert("RGB")
        # img1 = np.array(img, np.uint8 , copy=True)
        # img1 = img1[:, ::-1, :]
        # img1 = Image.fromarray(np.uint8(img1))
        # mano_pose_pca_mean =  np.array(sample_info["pose_m"],dtype=np.float32).squeeze()
        # mano_betas = np.array(sample_info["mano_betas"],dtype=np.float32)
        # hand_joint_3d = np.array(sample_info["joint_3d"],dtype=np.float32).squeeze()
        # joints_uv =  np.array(sample_info["joint_2d"],dtype=np.float32).squeeze()
        # #前三维是全局旋转，后45维是pose ,后三维是平移
        # mano_pose_aa_mean = np.concatenate((mano_pose_pca_mean[0:3],np.matmul(mano_pose_pca_mean[3:48],np.array(mano_layerl.smpl_data["hands_components"])),mano_pose_pca_mean[48:]),axis=0)
        # #前三维是全局旋转，后45维是pose ,后十维度是shape
        # mano_param = np.concatenate((mano_pose_aa_mean[0:48],mano_betas)) 

        # th_vertsl, th_jtrl = mano_layerl(th_pose_coeffs=torch.from_numpy(mano_param[None,:48]).float(),th_betas=torch.from_numpy(mano_param[None,48:]).float(),th_trans = torch.from_numpy(mano_pose_aa_mean[None,48:]).float())
        # mano_pose_aa_mean_wo_trans = mano_pose_aa_mean[:48].reshape(-1,3)
        # mano_pose_aa_mean_wo_trans[:,1:]  *= -1
        # mano_param = np.concatenate((mano_pose_aa_mean_wo_trans.reshape(-1),mano_betas)) 
        # th_vertsr, th_jtrr = mano_layer(th_pose_coeffs=torch.from_numpy(mano_param[None,:48]).float(),th_betas=torch.from_numpy(mano_param[None,48:]).float(),th_trans = torch.from_numpy(np.concatenate((mano_pose_aa_mean[None,48:49]*-1,mano_pose_aa_mean[None,49:]),axis=1)).float())
        # from PIL import ImageDraw
        # _,joints_hand_2d = dex_ycb_util.projectPoints((th_vertsl/1000).cpu().numpy().squeeze(),K)
        # draw = ImageDraw.Draw(img)

        # draw.point([tuple(i) for i in joints_hand_2d.tolist() ])
        # #draw.point([tuple(i) for i in joints_uv.tolist() ],(255,0,0))
        # img.save("./test1.jpg")

        # draw1 = ImageDraw.Draw(img1)
        # _,flip_joints_hand_2d = dex_ycb_util.projectPoints((th_vertsr/1000).cpu().numpy().squeeze(),K)
        # draw1.point([tuple(i) for i in flip_joints_hand_2d.tolist() ])
        # img1.save("./test2.jpg")



########################################################################################################
        # #2D物体可视化
        # sample_info = self.sample_dict[self.sample_list[5999]]
        # fx = sample_info['intrinsics']['fx']
        # fy = sample_info['intrinsics']['fy']
        # cx = sample_info['intrinsics']['ppx']
        # cy = sample_info['intrinsics']['ppy']
        # #camintr
        # K = np.zeros((3,3))
        # K[0,0] = fx
        # K[1,1] = fy
        # K[0,2] = cx
        # K[1,2] = cy
        # K[2,2] = 1
        # img =Image.open(os.path.join(self.root,sample_info["color_file"])).convert("RGB")
        # img1 = np.array(img, np.uint8 , copy=True)
        # img1 = img1[:, ::-1, :]
        # img1 = Image.fromarray(np.uint8(img1))

        # from PIL import ImageDraw

        # grasp_object_pose = np.array(sample_info["pose_y"][sample_info['ycb_grasp_ind']],dtype=np.float32)

        # _,p2d= dex_ycb_util.projectPoints(self.obj_bbox3d[sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]],K,rt=grasp_object_pose)


        # draw = ImageDraw.Draw(img)

        # draw.point([tuple(i) for i in p2d.tolist() ])
        # #draw.point([tuple(i) for i in joints_uv.tolist() ],(255,0,0))
        # img.save("./test1.jpg")


        # p2d[:,0] = np.array(img.size[0],dtype=np.float32)  - p2d[:,0] - 1

        # draw1 = ImageDraw.Draw(img1)
        
        # draw1.point([tuple(i) for i in p2d.tolist() ])
        # img1.save("./test2.jpg")