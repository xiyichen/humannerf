import os

import numpy as np
import pickle

import smplx
import torch

# MALE_PATH    = "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
# FEMALE_PATH  = "basicModel_f_lbs_10_207_0_v1.0.0.pkl"
# NEUTRAL_PATH = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


class SMPLX():
    def __init__(self, sex, model_dir):
        super(SMPLX, self).__init__()
        self.sex = sex
        self.model_dir = model_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_params = dict(model_type='smplx',
                        model_path=model_dir,
                        gender=sex,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        device=self.device,
                        dtype=torch.float64)
        self.model = smplx.create(**model_params)

    def __call__(self, pose, beta, global_orient=None):
        o = None
        if global_orient is not None:
            o = torch.tensor(global_orient, device=self.device).double().reshape(1, -1)
        body_model_output = self.model(return_verts=True,
                                       global_orient=o,
                                       body_pose=torch.tensor(pose, device=self.device).double().reshape(1, -1),
                                       betas=torch.tensor(beta, device=self.device).double().reshape(1, -1))
        v = body_model_output.vertices.squeeze(0).detach().cpu().numpy()
        joints = body_model_output.joints.squeeze(0).detach().cpu().numpy()[:22, :]

        return v, joints
