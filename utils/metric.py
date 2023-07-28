import numpy as np
import os
import cv2
import torch
# object evaluation metric
# hand evaluation metric: https://github.com/shreyashampali/ho3d/blob/master/eval.py


def vertices_reprojection(vertices, rt, k):
    p = np.matmul(k, np.matmul(rt[:3, 0:3], vertices.T) + rt[:3, 3].reshape(-1, 1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def compute_ADD_s_error(pred_pose, gt_pose, obj_mesh):
    #rot_dir_z = gt_pose[:3, 2] * np.pi  # N x 3
    #flipped_obj_rot_z = np.matmul(cv2.Rodrigues(rot_dir_z)[0].squeeze(),
    #                                        gt_pose[:3, 0:3])  # 3 x 3 # flipped rot
    N = obj_mesh.shape[0]
                                            
    add_gt = np.matmul(gt_pose[:3, 0:3], obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N)
    add_gt = torch.from_numpy(add_gt.T).cuda()
    #add_gt = add_gt[None,:,:].repeat(N,1)
    add_gt = add_gt.unsqueeze(0).repeat(N,1,1)
    #add_gt_flip = np.matmul(flipped_obj_rot_z, obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N)
    add_pred = np.matmul(pred_pose[:3, 0:3], obj_mesh.T) + pred_pose[:3, 3].reshape(-1, 1)
    add_pred = torch.from_numpy(add_pred.T).cuda()
    add_pred = add_pred.unsqueeze(1).repeat(1,N,1)
    #is_rot_sym_objs_z = cls in [6,21,10] #mustard, bleach, potted meat
    dis = torch.norm(add_gt - add_pred, dim=2)
    add_bias = torch.mean(torch.min(dis,dim=1)[0])
    add_bias = add_bias.detach().cpu().numpy()
    #add_bias = min(np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0),np.mean(np.linalg.norm(add_gt_flip - add_pred, axis=0), axis=0))
    #add_bias = np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0)
    return add_bias

def compute_REP_error(pred_pose, gt_pose, intrinsics, obj_mesh):
    reproj_pred = vertices_reprojection(obj_mesh, pred_pose, intrinsics)
    reproj_gt = vertices_reprojection(obj_mesh, gt_pose, intrinsics)
    reproj_diff = np.abs(reproj_gt - reproj_pred)
    reproj_bias = np.mean(np.linalg.norm(reproj_diff, axis=1), axis=0)
    return reproj_bias


def compute_ADD_error(pred_pose, gt_pose, obj_mesh):
    add_gt = np.matmul(gt_pose[:3, 0:3], obj_mesh.T) + gt_pose[:3, 3].reshape(-1, 1)  # (3,N) 
    add_pred = np.matmul(pred_pose[:3, 0:3], obj_mesh.T) + pred_pose[:3, 3].reshape(-1, 1)
    add_bias = np.mean(np.linalg.norm(add_gt - add_pred, axis=0), axis=0)
    return add_bias


def fuse_test(output, width, height, intrinsics, bestCnt, bbox_3d, cord_upleft, affinetrans=None,hand_type=None):
    predx = output[0]
    predy = output[1]
    det_confs = output[2]
    keypoints = bbox_3d
    nH, nW, nV = predx.shape

    xs = predx.reshape(nH * nW, -1) * width
    ys = predy.reshape(nH * nW, -1) * height
    det_confs = det_confs.reshape(nH * nW, -1)
    gridCnt = len(xs)

    p2d = None
    p3d = None
    candiBestCnt = min(gridCnt, bestCnt)
    for i in range(candiBestCnt):
        bestGrids = det_confs.argmax(axis=0) # choose best N count
        validmask = (det_confs[bestGrids, list(range(nV))] > 0.5)
        xsb = xs[bestGrids, list(range(nV))][validmask]
        ysb = ys[bestGrids, list(range(nV))][validmask]
        t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
        t3d = keypoints[validmask]
        if p2d is None:
            p2d = t2d
            p3d = t3d
        else:
            p2d = np.concatenate((p2d, t2d), 0)
            p3d = np.concatenate((p3d, t3d), 0)
        det_confs[bestGrids, list(range(nV))] = 0

    if len(p3d) < 6:
        R = np.eye(3)
        T = np.array([0, 0, 1]).reshape(-1, 1)
        rt = np.concatenate((R, T), 1)
        return rt, p2d

    p2d[:, 0] += cord_upleft[0]
    p2d[:, 1] += cord_upleft[1]
    if affinetrans is not None:
        affinetrans = np.linalg.inv(affinetrans)
        homp2d = np.concatenate([p2d, np.ones([np.array(p2d).shape[0], 1])], 1)
        p2d = affinetrans.dot(homp2d.transpose()).transpose()[:, :2]
    if hand_type == "left":
        p2d[:,0] = np.array(640,dtype=np.float32)  - p2d[:,0] - 1
    retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
    if not retval:
        R = np.eye(3)
        T = np.array([0, 0, 1]).reshape(-1, 1)
    else:
        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
    rt = np.concatenate((R, T), 1)
    return rt, p2d


def eval_batch_obj(batch_output, obj_bbox,
                   obj_pose, mesh_dict, obj_bbox3d, obj_cls,
                   cam_intr, REP_res_dic, ADD_res_dic, bestCnt=10, batch_affinetrans=None,batch_hand_type=None):
    # bestCnt: choose best N count for fusion
    bs = batch_output[0].shape[0]
    obj_bbox = obj_bbox.cpu().numpy()
    for i in range(bs):
        output = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        bbox = obj_bbox[i]
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cord_upleft = [bbox[0], bbox[1]]
        intrinsics = cam_intr[i]
        bbox_3d = obj_bbox3d[i]
        if torch.is_tensor(obj_cls[i]):
            cls = int(obj_cls[i])
        else:
            cls = obj_cls[i]
        mesh = mesh_dict[cls]
        hand_type=batch_hand_type[i]
        if batch_affinetrans is not None:
            affinetrans = batch_affinetrans[i]
        else:
            affinetrans = None
        pred_pose, p2d = fuse_test(output, width, height, intrinsics, bestCnt, bbox_3d, cord_upleft,
                                   affinetrans=affinetrans,hand_type=hand_type)
        # calculate REP and ADD error
        REP_error = compute_REP_error(pred_pose, obj_pose[i], intrinsics, mesh)
        if cls in [13,16,20,21]:
            ADD_error = compute_ADD_s_error(pred_pose, obj_pose[i], mesh)
        else:
            ADD_error = compute_ADD_error(pred_pose, obj_pose[i], mesh)
        REP_res_dic[cls].append(REP_error)
        ADD_res_dic[cls].append(ADD_error)
    return REP_res_dic, ADD_res_dic


def eval_object_pose(REP_res_dic, ADD_res_dic, diameter_dic, outpath, unseen_objects=[], epoch=None):
    # REP_res_dic: key: object class, value: REP error distance
    # ADD_res_dic: key: object class, value: ADD error distance

    # object result file
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    log_path = os.path.join(outpath, "object_result.txt") if epoch is None else os.path.join(outpath, "object_result_epoch{}.txt".format(epoch))
    log_file = open(log_path, "w+")

    REP_5 = {}
    for k in REP_res_dic.keys():
        REP_5[k] = np.mean(np.array(REP_res_dic[k]) <= 5)

    ADD_10 = {}
    for k in ADD_res_dic.keys():
        ADD_10[k] = np.mean(np.array(ADD_res_dic[k]) <= 0.1 * diameter_dic[k])

    # for k in ADD_res_dic.keys():
    #     if k in unseen_objects:
    #         REP_5.pop(k, None)
    #         ADD_10.pop(k, None)

    # write down result
    print('REP-5', file=log_file)
    print(REP_5, file=log_file)
    print('ADD-10', file=log_file)
    print(ADD_10, file=log_file)
    log_file.close()
    return ADD_10, REP_5

def eval_hand_pose_result(hand_eval_result,outpath, epoch):
    #hand result file
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    log_path = os.path.join(outpath, "hand_result.txt") if epoch is None else os.path.join(outpath, "hand_result_epoch{}.txt".format(epoch))
    log_file = open(log_path, "w+")
    print('mpjpe', file=log_file)
    print(np.mean(np.array(hand_eval_result[0])), file=log_file)
    print('pa-mpjpe', file=log_file)
    print(np.mean(np.array(hand_eval_result[1])), file=log_file)

    log_file.close()


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A,B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def eval_hand(preds_joint,gts_root_joint,gts_hand_type,gts_joints_coord_cam,hand_eval_result):
    sample_num = len(preds_joint)
    for n in range (sample_num):
        pred_joint = preds_joint[n]
        gt_hand_type = gts_hand_type[n]
        gt_root_joint = gts_root_joint[n].detach().cpu().numpy()
        gt_joints_coord_cam = gts_joints_coord_cam[n].detach().cpu().numpy()

        # root centered
        #\u5df2\u5728mano_ho3d\u5c42\u505a\u8fc7

        # flip back to left hand
        if gt_hand_type == 'left':
            pred_joint[:,0] *= -1

        # root align
        pred_joint += gt_root_joint

        # GT and rigid align
        joints_out_aligned = rigid_align(pred_joint, gt_joints_coord_cam)

        #m to mm
        pred_joint *= 1000
        joints_out_aligned *= 1000
        gt_joints_coord_cam *= 1000

            
        #[mpjpe_list, pa-mpjpe_list]
        hand_eval_result[0].append(np.sqrt(np.sum((pred_joint - gt_joints_coord_cam)**2,1)).mean())
        hand_eval_result[1].append(np.sqrt(np.sum((joints_out_aligned - gt_joints_coord_cam)**2,1)).mean())

    return hand_eval_result



        