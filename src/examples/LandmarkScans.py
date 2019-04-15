import sys
sys.path.append('../')

import os
import morf.landmark as lm
import morf.utils as utils

scan_dir = './scans/subjects'
landmark_dir = './scans/landmarks'

for filename in os.listdir(scan_dir):
    infile = os.path.join(scan_dir, filename)

    outfile = os.path.join(landmark_dir, filename[:-3]+"txt")

    mesh = utils.read_mesh(infile)
    landmarks = lm.identify_3D_landmarks(mesh)
    landmarks = [landmarks[x] for x in [27,30,33,62,8,36,39,42,45,48,54]]

    utils.save_landmarks(landmarks, outfile)

