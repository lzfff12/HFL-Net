import torch
from torch import nn
from torchvision import ops
from networks.backbone_share import FPN
from networks.hand_head import hand_Encoder, hand_regHead
from networks.object_head import obj_regHead, Pose2DLayer
from networks.mano_head import mano_regHead
from networks.CR import Transformer
from networks.loss import Joint2DLoss, ManoLoss, ObjectLoss

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

class HONet(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HONet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head)
        self.transformer_hand = Transformer(inp_res=roi_res, dim=channels*2,depth=transformer_depth, num_heads=transformer_head)

        self.hand_head.apply(init_weights)
        self.hand_encoder.apply(init_weights)
        self.mano_branch.apply(init_weights)
        self.transformer_obj.apply(init_weights)
        self.transformer_hand.apply(init_weights)
        self.obj_head.apply(init_weights)



    def net_forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None):
        batch = self.new_method(imgs)

        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2
        # P2 from FPN Network
        P2_h,P2_o = self.base_net(imgs)
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x_hand = ops.roi_align(P2_h, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        x_obj = ops.roi_align(P2_o, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand

        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2_o, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj



            z_x = ops.roi_align(P2_h, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection

    
            z_x = msk_inter[:, None, None, None] * z_x

            #print(3)

            hand_obj = torch.cat([x_hand,x_obj.detach()],dim=1)
            hand_obj = self.transformer_hand(hand_obj,hand_obj)

            y = self.transformer_obj(y, z_x.detach())

            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None

        hand = hand_obj[:,0:256,:,:]
        #hand forward

        out_hm, encoding, preds_joints = self.hand_head(hand)

        mano_encoding = self.hand_encoder(out_hm, encoding)

        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params, roots3d=roots3d)

        return preds_joints, pred_mano_results, gt_mano_results, preds_obj

    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None):
        if self.training:
            preds_joints, pred_mano_results, gt_mano_results, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                                           mano_params=mano_params)
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj
        else:
            preds_joints, pred_mano_results, _, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                             roots3d=roots3d)
            return preds_joints, pred_mano_results, preds_obj


class HOModel(nn.Module):

    def __init__(self, honet, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None):

        super(HOModel, self).__init__()
        self.honet = honet
        # supervise when provide mano params
        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        # supervise when provide hand joints
        self.mano_joint_loss = ManoLoss(lambda_joints3d=mano_lambda_joints3d,
                                        lambda_regulshape=mano_lambda_regulshape,
                                        lambda_regulpose=mano_lambda_regulpose)
        # object loss
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

    def forward(self, imgs, bbox_hand, bbox_obj,
                joints_uv=None, joints_xyz=None, mano_params=None, roots3d=None,
                obj_p2d_gt=None, obj_mask=None, obj_lossmask=None):
        if self.training:
            losses = {}
            total_loss = 0
            preds_joints2d, pred_mano_results, gt_mano_results, preds_obj= self.honet(imgs, bbox_hand, bbox_obj,mano_params=mano_params)
            if mano_params is not None:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(preds_joints2d, joints_uv)
                for key, val in joint2d_losses.items():
                    losses[key] = val
                total_loss += joint2d_loss
            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj, obj_lossmask=obj_lossmask)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0
            return total_loss, losses
        else:
            preds_joints, pred_mano_results, _, preds_obj = self.honet.module.net_forward(imgs, bbox_hand, bbox_obj, roots3d=roots3d)
            return preds_joints, pred_mano_results, preds_obj