import os
import sys
import cv2
import vtk
import numpy as np
import face_recognition as fr
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

lm_dir = "../CNNLM"

f = sys.argv[1]

def main():
    print(f)
    mesh = read_mesh(f)
    landmarks = compute_landmarks(mesh)
    outfile = os.path.join(lm_dir, f.split("/")[-1][:-3]+"txt")

    if(landmarks):
        save_landmarks(landmarks, outfile)


def compute_landmarks(mesh):
    landmark_ids = [36,39,27,42,45,30,33,60,51,64,57,8]

    # Find the coarse facial orientation in 3D
    c = _compute_centroid(mesh)
    d = 600
    camera_positions = [
        (c[0], c[1], c[2]+d),
        (c[0], c[1]+d, c[2]+d),
        (c[0]+d, c[1], c[2]+d),
        (c[0], c[1]-d, c[2]+d),
        (c[0]-d, c[1], c[2]+d)
    ]

    landmarks_3d = None
    for position in camera_positions:
        scene = Scene(mesh, position, c, 50, 500)
        image = scene.captureImage()

        if len(fr.face_landmarks(image)) > 0: 
            print("Face Located")
            landmarks_2d = _identify_2D_landmarks(image)
            #_check_image(image, landmarks_2d)
            landmarks_3d = [scene.pickPoint(point_2d) for point_2d in landmarks_2d]
            break

    if landmarks_3d == None:
        print("Face Not Found!")
        return

    # recompute the camera position for better landmarks
    for i in range(2):
        focal_point, camera_position, view_up = _compute_frontal_camera_settings(landmarks_3d, 800)
        scene2 = Scene(mesh, camera_position, focal_point, 20, 800, view_up)
        image = scene2.captureImage()
        #_check_image(image)

        landmarks_2d = _identify_2D_landmarks(image)
        landmarks_3d = [scene2.pickPoint(point_2d) for point_2d in landmarks_2d]

    #_check_image(image, landmarks_2d)
    return [landmarks_3d[x] for x in landmark_ids]



def _check_image(image, landmarks_2d=None):
    image = image.copy()

    if landmarks_2d:
        for point_2d in landmarks_2d:
            cv2.circle(image, point_2d, 2, (255,255,0))

    cv2.imshow('im', image)

    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def _compute_frontal_camera_settings(landmarks, face_distance):
    nose_bridge = np.asarray(landmarks[27])
    nose_tip = np.asarray(landmarks[30])
    left_eye = np.asarray(landmarks[45])
    left_lip = np.asarray(landmarks[54])
    right_eye = np.asarray(landmarks[36])
    right_lip = np.asarray(landmarks[48])
    ls = np.asarray(landmarks[51])


    v1 = np.subtract(right_lip, left_eye)
    v2 = np.subtract(left_lip, right_eye)

    direction = np.cross(v1, v2)
    direction = direction / np.linalg.norm(direction)

    focal_point = nose_tip
    camera_position = nose_tip + face_distance*direction
    view_up = nose_bridge - ls

    return focal_point, camera_position, view_up


def _identify_2D_landmarks(image):
    landmarks = fr.face_landmarks(image)[0]

    # unpack from the dictionary into the right order...
    landmark_list = landmarks['chin'] + landmarks['left_eyebrow'] + \
                    landmarks['right_eyebrow'] + landmarks['nose_bridge'] + \
                    landmarks['nose_tip'] + landmarks['left_eye'] + landmarks['right_eye'] + \
                    landmarks['top_lip'][:-5] + landmarks['bottom_lip'][1:-6] + \
                    landmarks['top_lip'][:-5:-1] + landmarks['bottom_lip'][:-5:-1]

    return landmark_list




def vtkImage_to_np(image):
    rows, cols, _ = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()

    np_array = vtk_to_numpy(scalars)
    np_array = np_array.reshape(rows, cols, -1)

    # vtk and cv2 use different colorspaces...
    red, green, blue = np.dsplit(np_array, np_array.shape[-1])
    np_array = np.stack([blue, green, red], 2).squeeze()

    # the first axis of the image is also flipped...
    np_array = np.flip(np_array, 0)

    return np_array

def compute_centroid(mesh):
    com = vtk.vtkCenterOfMass()
    com.SetInputData(mesh)
    com.Update()
    return com.GetCenter()


def save_landmarks(landmarks, filename):
    np.savetxt(filename, landmarks)


def read_mesh(filename):
    file_type = filename[-3:]

    if file_type == 'ply':
        reader = vtk.vtkPLYReader()
        
    elif file_type == 'obj':
        reader = vtk.vtkOBJReader()

    elif file_type == 'stl':
        reader = vtk.vtkSTLReader()
        
    else:
        return None


    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()



main()
