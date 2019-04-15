import vtk
import numpy as np

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
    ids = vtk.vtkIdList()

    lines.InitTraversal()
    while lines.GetNextCell(ids):
        connections.append((ids.GetId(0), ids.GetId(1)))

    return connections


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
