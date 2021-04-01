import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def clean_mesh(mesh):
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh)
    cleaner.PointMergingOn()
    #cleaner.SetTolerance(0.0)
    cleaner.Update()
    return cleaner.GetOutput()

def triangulate_mesh(mesh):
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(mesh)
    triangulator.Update()
    return triangulator.GetOutput()

def connect_mesh_filter(mesh):
    connect_filter = vtk.vtkPolyDataConnectivityFilter()
    connect_filter.SetInputData(mesh)
    connect_filter.SetExtractionModeToLargestRegion()
    connect_filter.Update()
    return connect_filter.GetOutput()

def smooth_mesh(mesh):
    smoother = vtk.vtkSmoothPolyData()
    smoother.SetInputData(mesh)

    smoother.Update()
    return smoother.GetOutput()


def extract_connections(mesh):
    edges = vtk.vtkExtractEdges()
    edges.SetInputData(mesh)
    edges.Update()
    lines = edges.GetOutput().GetLines()

    connections = []
    edges_length = []
    ids = vtk.vtkIdList()

    lines.InitTraversal()
    while lines.GetNextCell(ids):
        p1_id = ids.GetId(0)
        p2_id = ids.GetId(1)
        points = vtk_to_numpy(mesh.GetPoints().GetData())
        connections.append((p1_id, p2_id))
        p1 = points[p1_id]
        p2 = points[p2_id]
        edge_length = np.linalg.norm(p1 - p2)
        edges_length.append(edge_length)
    edges_length = np.array(edges_length)
    edges_length_norm = edges_length / edges_length.mean()
    return connections, edges_length_norm


def save_landmarks(landmarks, filename):
    np.savetxt(filename, landmarks)


def read_landmarks(filename):
    return np.loadtxt(filename)


def save_mesh(mesh, filename):
    file_type = filename[-3:]

    if file_type == 'ply':
        saver = vtk.vtkPLYWriter()
        
    elif file_type == 'vtk':
        saver = vtk.vtkPolyDataWriter()

    else:
        print("Only ply and vtk file types are supported")
        return

    saver.SetInputData(mesh)
    saver.SetFileName(filename)
    saver.Update()


def read_mesh(filename):
    file_type = filename[-3:]

    if file_type == 'ply':
        reader = vtk.vtkPLYReader()
        
    elif file_type == 'obj':
        reader = vtk.vtkOBJReader()

    elif file_type == 'stl':
        reader = vtk.vtkSTLReader()
        
    elif file_type == 'vtk':
        reader = vtk.vtkPolyDataReader()

    else:
        print("Only ply, obj, vtk, or stl file types are supported")
        return 

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def np_to_vtkPoints(points):
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])
    vtk_points.Modified()
    return vtk_points


def vtkPoints_to_np(points):
    return vtk.util.numpy_support.vtk_to_numpy(points.GetData())


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
