import sys
sys.path.append('../')

import os
import morf.landmark as lm
import morf.utils as utils

scan_dir = './data/subjects'
landmark_dir = './data/landmarks'

for filename in os.listdir(scan_dir):

    infile = os.path.join(scan_dir, filename)
    outfile = os.path.join(landmark_dir, filename[:-3]+"txt")
    mesh = utils.read_mesh(infile)

    landmarks = lm.identify_3D_landmarks(mesh)
    utils.save_landmarks(landmarks, outfile)
