import sys
sys.path.append('../')

import os
import morf.landmark as lm
import morf.registration as reg
import morf.utils as utils

atlas_mesh_file = './data/atlas/atlas.ply'
atlas_landmark_file = './data/atlas/atlas_lm_12.txt'

mesh_file = './data/subjects/jordan.ply'
landmark_file = './data/landmarks/jordan.txt'

out_file = './data/registrations/jordan.ply'

atlas_mesh = utils.read_mesh(atlas_mesh_file)
atlas_landmarks = utils.read_landmarks(atlas_landmark_file)

mesh = utils.read_mesh(mesh_file)
mesh = utils.clean_mesh(mesh)
landmarks = utils.read_landmarks(landmark_file)

lm._check_landmarks_3d(mesh, landmarks)
lm._check_landmarks_3d(atlas_mesh, atlas_landmarks)

# Registration
atlas_mesh, atlas_landmarks = reg.affine_alignment(atlas_mesh, atlas_landmarks, landmarks)
atlas_mesh = reg.spline(atlas_mesh, atlas_landmarks, landmarks)

registered_mesh = reg.non_rigid_icp(
        atlas_mesh, 
        mesh,
        max_stiffness=10000,
        min_stiffness=50,
        max_iterations=50
        )

lm._check_landmarks_3d(registered_mesh, landmarks)
utils.save_mesh(registered_mesh, out_file)
