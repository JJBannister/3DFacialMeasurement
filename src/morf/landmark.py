import os
import time
import cv2
import vtk
import numpy as np
import face_recognition as fr

from . import utils

_image_size = 1000
_face_distance = 700
_landmark_ids = [36,39,27,42,45,30,33,60,51,64,57,8,62,66,31,35]


def identify_3D_landmarks(mesh, visualize = True):
    """
    Returns a set of 3D facial landmarks on the mesh

    :param mesh: vtkPolyData polygonal mesh
    :return: A np array of 3D points
    """

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
            landmarks_2d = identify_2D_landmarks(image)
            if visualize:
                _check_landmarks_2d(image, landmarks_2d)
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
        #_check_landmarks_2d(image)

        landmarks_2d = identify_2D_landmarks(image)
        landmarks_3d = [scene2.pickPoint(point_2d) for point_2d in landmarks_2d]

    if visualize:
        _check_landmarks_2d(image, landmarks_2d)
        _check_landmarks_3d(mesh, landmarks_3d)
    return [landmarks_3d[x] for x in _landmark_ids]


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
    def __init__(self, mesh, camera_position, focal_point, view_angle, image_size, view_up = (0,1,0)):
        self.graphics = vtk.vtkGraphicsFactory()
        self.graphics.SetUseMesaClasses(1)
        #self.img_factory = vtk.vtkImagingFactory()
        #self.img_factory.SetUseMesaClasses(1)

        self.mesh = mesh
        self.image_size = image_size

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.mesh)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.camera = vtk.vtkCamera()

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetActiveCamera(self.camera)
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0,0,0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(True)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.picker = vtk.vtkPointPicker()
        self.interactor.SetPicker(self.picker)

        self.render_window.SetSize(self.image_size, self.image_size)
        self.camera.SetPosition(camera_position[0], camera_position[1], camera_position[2])
        self.camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
        self.camera.SetViewUp(view_up)
        self.camera.OrthogonalizeViewUp()
        self.camera.SetViewAngle(view_angle)

        self.image_filter = vtk.vtkWindowToImageFilter()
        self.image_filter.SetInput(self.render_window)


    def render(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()


    def captureImage(self):
        self.render_window.Render()
        self.image_filter.Update()
        return utils.vtkImage_to_np(self.image_filter.GetOutput())


    def pickPoint(self, point_2d):
        self.render_window.Render()

        # the second axis of the image is flipped...
        point_2d = (point_2d[0], self.image_size - point_2d[1])

        self.picker.Pick(point_2d[0], point_2d[1], 0, self.renderer)
        point_id = self.picker.GetPointId()

        if point_id >=0 and point_id < self.mesh.GetNumberOfPoints():
            point_3d = self.mesh.GetPoints().GetPoint(point_id)
        else:
            point_3d = None

        return point_3d


def _check_landmarks_2d(image, landmarks_2d=None):
    image = image.copy()

    if landmarks_2d:
        for point_2d in landmarks_2d:
            cv2.circle(image, point_2d, 2, (255,255,0))

    cv2.imshow('im', image)

    cv2.waitKey(3000)
    cv2.destroyAllWindows()


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


def _compute_centroid(mesh):
    com = vtk.vtkCenterOfMass()
    com.SetInputData(mesh)
    com.Update()

    return com.GetCenter()
    

