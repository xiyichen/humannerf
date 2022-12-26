import os
import sys

import json
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))
from third_parties.smplx.smplx_numpy import SMPLX
from third_parties.smpl.smpl_numpy import SMPL
import cv2

from absl import app
from absl import flags
import glob
import joblib
import torch

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    '''  same as batch_matrix2axis
    Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    Code from smplx/flame, what PS people often use
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'wild.yaml',
                    'the path of config file')

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config

def convert_jpg_to_png(image_path):
    print('converting jpg to png')
    for fname in tqdm(glob.glob(image_path + '/*')):
        im = cv2.imread(fname)
        cv2.imwrite(os.path.join(image_path, fname.split('/')[-1].split('.')[0] + '.png'), im)

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def batch_rot2aa(Rs):
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);
        out_theta = std::acos(c);
        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

def prepare_metadata(dataset_path, USE_SMPLX, Fitting_method):
    model_name = 'smplx' if USE_SMPLX else 'smpl'
    print('preparing metadata from ' + model_name + ' fittings')
    full_dict = {}
    for fname in tqdm(glob.glob(os.path.join(dataset_path, 'images', '*.png'))):
        frame_name = fname.split('/')[-1].split('.')[0]
        frame_smpl_dict = joblib.load(os.path.join(dataset_path, model_name + '_fittings', frame_name + '.pkl'))
        if Fitting_method == 'pare':
            poses_rotmat = frame_smpl_dict['pred_pose'].squeeze(0)
            poses = batch_rot2aa(torch.tensor(poses_rotmat)).reshape(1, -1)
            betas = frame_smpl_dict['pred_shape']
            FOCAL_LENGTH = 5000
            pred_cam = frame_smpl_dict['pred_cam'][0]
            bbox = frame_smpl_dict['bboxes'][0]
            CROP_SIZE = 224

            bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
            assert bbox_w == bbox_h

            bbox_size = bbox_w
            bbox_x = bbox_cx - bbox_w / 2.
            bbox_y = bbox_cy - bbox_h / 2.

            scale = bbox_size / CROP_SIZE
            cam_intrinsics = np.eye(3)
            cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
            cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
            cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
            cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y
            cam_s, cam_tx, cam_ty = pred_cam
            trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

            cam_extrinsics = np.eye(4)
            cam_extrinsics[:3, 3] = trans
        elif Fitting_method == 'smplify-x-partial':
            poses = np.append(frame_smpl_dict['global_orient'][0], frame_smpl_dict['body_pose'][0]).reshape(-1, 3)
            # poses = batch_rodrigues(torch.tensor(poses))
            poses = quaternion_to_rotation_matrix(euler_to_quaternion(torch.tensor(poses)))
            poses = batch_rot2aa(poses).detach().cpu().numpy().reshape(1, -1)
            betas = frame_smpl_dict['betas']
            focal_length = frame_smpl_dict['focal_length']
            camera_center = frame_smpl_dict['camera_center']
            camera_translation = frame_smpl_dict['camera_translation']
            cam_intrinsics = np.eye(3)
            cam_intrinsics[0, 0] = focal_length
            cam_intrinsics[1, 1] = focal_length
            cam_intrinsics[:2, 2] = camera_center[0]
            cam_extrinsics = np.eye(4)
            cam_extrinsics[:3, 3] = camera_translation[0]
        else:
            raise Exception('Unsupported fitting method {}, please add corresponding pkl parser'.format(Fitting_method))
        dict_frame = {}
        dict_frame['poses'] = poses[0]
        dict_frame['betas'] = betas[0]
        dict_frame['cam_intrinsics'] = cam_intrinsics
        dict_frame['cam_extrinsics'] = cam_extrinsics
        full_dict[frame_name] = dict_frame
        for k in dict_frame:
            dict_frame[k] = dict_frame[k].tolist()
    with open(os.path.join(dataset_path, 'metadata.json'), "w") as outfile:
        json.dump(full_dict, outfile, indent=4, cls=NumpyFloatValuesEncoder)


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']

    dataset_dir = cfg['dataset']['path']
    subject_dir = os.path.join(dataset_dir, subject)
    USE_SMPLX = cfg['dataset']['use_smplx']
    Fitting_method = cfg['dataset']['fitting_method']
    if USE_SMPLX:
        print('Using SMPL-X')
        MODEL_DIR = '../../third_parties/smplx/models/'
    else:
        print('Using SMPL')
        MODEL_DIR = '../../third_parties/smpl/models/'
    output_path = subject_dir

    if glob.glob(os.path.join(subject_dir, 'images/*.png')) == []:
        convert_jpg_to_png(os.path.join(subject_dir, 'images'))
    
    # if not os.path.isfile(os.path.join(subject_dir, 'metadata.json')):
    prepare_metadata(subject_dir, USE_SMPLX, Fitting_method)
    
    with open(os.path.join(subject_dir, 'metadata.json'), 'r') as f:
        frame_infos = json.load(f)

    if USE_SMPLX:
        model = SMPLX(sex=sex, model_dir=MODEL_DIR)
    else:
        model = SMPL(sex=sex, model_dir=MODEL_DIR)


    cameras = {}
    mesh_infos = {}
    all_betas = []
    print('preparing dataset')
    for frame_base_name in tqdm(frame_infos):
        cam_body_info = frame_infos[frame_base_name] 
        poses = np.array(cam_body_info['poses'], dtype=np.float32)
        betas = np.array(cam_body_info['betas'], dtype=np.float32)
        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)
        
        all_betas.append(betas)

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        vertices_smplx_space = model(poses, betas)[0]
        _, tpose_joints = model(np.zeros_like(poses), betas)

        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        Rh = poses[:3].copy()

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose
        poses[:3] = 0

        # get posed joints using body poses without global rotation
        vertices, joints = model(poses, betas)
        joints = joints - pelvis_pos[None, :]
        vertices = vertices - pelvis_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints,
            # 'vertices_smplx_space': vertices_smplx_space,
            # 'vertices': vertices,
            'tpose_joints': tpose_joints
        }

        cameras[frame_base_name] = {
            'intrinsics': K,
            'extrinsics': E
        }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    if USE_SMPLX:
        model_canonical = SMPLX(sex=sex, model_dir=MODEL_DIR)
    else:
        model_canonical = SMPL(sex=sex, model_dir=MODEL_DIR)
    _, template_joints = model_canonical(np.zeros(poses.shape[0]), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)


if __name__ == '__main__':
    app.run(main)