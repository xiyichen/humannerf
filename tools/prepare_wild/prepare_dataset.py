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

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'wild.yaml',
                    'the path of config file')
USE_SMPLX = True
if USE_SMPLX:
    print('Using SMPL-X')
    MODEL_DIR = '../../third_parties/smplx/models'
else:
    print('Using SMPL')
    MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config

def convert_jpg_to_png(image_path):
    print('converting jpg to png')
    for fname in tqdm(glob.glob(image_path + '/*')):
        im = cv2.imread(fname)
        cv2.imwrite(fname.split('.')[0] + '.png', im)

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def prepare_metadata(dataset_path):
    model_name = 'smplx' if USE_SMPLX else 'smpl'
    print('preparing metadata from ' + model_name + ' fittings')
    full_dict = {}
    for fname in tqdm(glob.glob(os.path.join(dataset_path, 'images', '*.png'))):
        frame_name = fname.split('/')[-1].split('.')[0]
        frame_smpl_dict = joblib.load(os.path.join(dataset_path, model_name + '_fittings', frame_name + '.pkl'))
        poses = frame_smpl_dict['body_pose']
        betas = frame_smpl_dict['betas']
        if model_name == 'smplx':
            global_orient = frame_smpl_dict['global_orient']
        focal_length = frame_smpl_dict['focal_length']
        camera_center = frame_smpl_dict['camera_center']
        camera_translation = frame_smpl_dict['camera_translation']
        cam_k = np.eye(3)
        cam_k[0, 0] = focal_length
        cam_k[1, 1] = focal_length
        cam_k[:2, 2] = camera_center[0]
        cam_e = np.eye(4)
        cam_e[:3, 3] = camera_translation[0]
        dict_frame = {}
        dict_frame['poses'] = poses[0]
        dict_frame['betas'] = betas[0]
        if model_name == 'smplx':
            dict_frame['global_orient'] = global_orient[0]
        dict_frame['cam_intrinsics'] = cam_k
        dict_frame['cam_extrinsics'] = cam_e
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
    output_path = subject_dir

    if glob.glob(os.path.join(subject_dir, 'images/*.png')) == []:
        convert_jpg_to_png(os.path.join(subject_dir, 'images'))
    
    if not os.path.isfile(os.path.join(subject_dir, 'metadata.json')):
        prepare_metadata(subject_dir)
    
    with open(os.path.join(subject_dir, 'metadata.json'), 'r') as f:
        frame_infos = json.load(f)

    if USE_SMPLX:
        model = SMPLX(sex=sex, model_dir=MODEL_DIR)
    else:
        model = SMPL(sex=sex, model_dir=MODEL_DIR)


    cameras = {}
    mesh_infos = {}
    all_betas = []
    for frame_base_name in tqdm(frame_infos):
        cam_body_info = frame_infos[frame_base_name] 
        poses = np.array(cam_body_info['poses'], dtype=np.float32)
        betas = np.array(cam_body_info['betas'], dtype=np.float32)
        if 'global_orient' in cam_body_info:
            global_orient = np.array(cam_body_info['global_orient'], dtype=np.float32)
        else:
            global_orient = None
        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)
        
        all_betas.append(betas)

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, tpose_joints = model(np.zeros_like(poses), betas)

        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        if USE_SMPLX:
            Rh = global_orient
        else:
            Rh = poses[:3].copy()
            poses[:3] = 0

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose

        # get posed joints using body poses without global rotation
        _, joints = model(poses, betas, global_orient=global_orient)
        joints = joints - pelvis_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': np.append(np.zeros(3), poses) if USE_SMPLX else poses,
            'joints': joints,
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
