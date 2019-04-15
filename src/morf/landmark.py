import os
import time
import cv2
import vtk
import numpy as np
import face_recognition as fr
from vtk.util.numpy_support import vtk_to_numpy

from . import utils

_image_size = 1000
_face_distance = 700


def identify_3D_landmarks(mesh):
    # initial guess of facial position/orientation
    focal_point = _compute_centroid(mesh)
    camera_position = (focal_point[0], focal_point[1], focal_point[2]+_face_distance)
    view_angle = 70

    scene = Scene(mesh, camera_position, focal_point, view_angle)
    image = scene.captureImage()

    landmarks_2d = identify_2D_landmarks(image)
    landmarks_3d = [scene.pickPoint(point_2d) for point_2d in landmarks_2d]

    # recompute the camera position for better landmarks
    focal_point, camera_position = _compute_camera(landmarks_3d)
    view_angle = 20
    scene = Scene(mesh, camera_position, focal_point, view_angle)
    image = scene.captureImage()

    landmarks_2d = identify_2D_landmarks(image)

    landmarks_3d = [scene.pickPoint(point_2d) for point_2d in landmarks_2d]

    return landmarks_3d


def identify_2D_landmarks(image):
    landmarks = fr.face_landmarks(image)[0]

    # unpack from the dictionary into the right order...
    landmark_list = landmarks['chin'] + landmarks['left_eyebrow'] + \
        landmarks['right_eyebrow'] + landmarks['nose_bridge'] + \
        landmarks['nose_tip'] + landmarks['left_eye'] + landmarks['right_eye'] + \
        landmarks['top_lip'][:-5] + landmarks['bottom_lip'][1:-6] + \
        landmarks['top_lip'][:-5:-1] + landmarks['bottom_lip'][:-5:-1]

    return landmark_list


class Scene:
    def __init__(self, mesh, camera_position, focal_point, view_angle):
        self.mesh = mesh

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.mesh)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.camera = vtk.vtkCamera()

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetActiveCamera(self.camera)
        self.renderer.AddActor(self.actor)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.picker = vtk.vtkPointPicker()
        self.interactor.SetPicker(self.picker)

        self.render_window.SetSize(_image_size,_image_size)
        self.camera.SetPosition(camera_position[0], camera_position[1], camera_position[2])
        self.camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
        self.camera.SetViewAngle(view_angle)

        self.image_filter = vtk.vtkWindowToImageFilter()
        self.image_filter.SetInput(self.render_window)


    def render(self):
        self.render_window.SetOffScreenRendering(0)
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()
        self.render_window.SetOffScreenRendering(1)


    def captureImage(self):
        #self.interactor.Initialize()
        self.render_window.Render()

        self.image_filter.Update()
        return _vtk_to_np(self.image_filter.GetOutput())


    def pickPoint(self, point_2d):
        #self.interactor.Initialize()
        self.render_window.Render()

        # the second axis of the image is flipped...
        point_2d = (point_2d[0], _image_size - point_2d[1])

        self.picker.Pick(point_2d[0], point_2d[1], 0, self.renderer)
        point_id = self.picker.GetPointId()
        point_3d = self.mesh.GetPoints().GetPoint(point_id)

        return point_3d


def _check_landmarks_2d(image, landmarks_2d):
    image = image.copy()

    for point_2d in landmarks_2d:
        cv2.circle(image, point_2d, 2, (0,0,255))

    cv2.imshow('im', image)
    cv2.waitKey(0)


def _check_landmarks_3d(mesh, landmarks_3d):
    points = utils.np_to_vtkPoints(landmarks_3d)

    landmarks = vtk.vtkPolyData()
    landmarks.SetPoints(points)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(landmarks)

    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(glyph.GetOutputPort())

    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(sphere_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(render_window)
    renderWindowInteractor.Initialize()

    render_window.Render()
    renderWindowInteractor.Start()


def _compute_camera(landmarks):
    nose_bridge = np.asarray(landmarks[27])
    nose_tip = np.asarray(landmarks[30])
    left_eye = np.asarray(landmarks[45])
    left_lip = np.asarray(landmarks[54])
    right_eye = np.asarray(landmarks[36])
    right_lip = np.asarray(landmarks[48])
    chin = np.asarray(landmarks[8])

    v1 = np.subtract(right_lip, nose_bridge)
    v2 = np.subtract(left_lip, nose_bridge)

    direction = np.cross(v1, v2)
    direction = direction / np.linalg.norm(direction)

    focal_point = nose_tip
    camera_position = nose_tip + _face_distance*direction

    return focal_point, camera_position


def _compute_centroid(mesh):
    com = vtk.vtkCenterOfMass()
    com.SetInputData(mesh)
    com.Update()

    return com.GetCenter()
    

def _vtk_to_np(image):
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


