import time
import vtk
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, norm
import numpy as np
from . import utils


def non_rigid_icp(source_mesh, target_mesh,
                  max_iterations=50, norm_threshold=3,
                  max_stiffness=10000, min_stiffness=100, stiffness_step_factor=2,
                  verbose=True,
                  point_identifier = None):

    """
    Applies the n-icp algorithm to the source mesh.

    “Optimal Step Nonrigid ICP Algorithms for Surface Registration”
    B. Amberg, S. Romdhani, and T. Vetter,

    :param source_mesh: vtkPolyData polygonal mesh

    :param target_mesh: vtkPolyData polygonal mesh
    :return: the transformed source mesh
    """

    # Setup
    source_points = utils.vtkPoints_to_np(source_mesh.GetPoints())
    connections, edge_scale = utils.extract_connections(source_mesh)

    n_points = source_mesh.GetPoints().GetNumberOfPoints()
    n_connections = len(connections)

    if verbose:
        print()
        print("*** N-ICP ***")
        print("N src points: ", n_points)
        print("N src connections: ", n_connections)

    if point_identifier is None:
        point_identifier = PointIdentifier(target_mesh)

    stiff = max_stiffness
    X_old = None

    # Go
    for loop in range(max_iterations):
        if stiff < min_stiffness:
            break

        print()
        print("loop: ", loop)
        start = time.time()

        # Solve AX = B for transformations X
        A = lil_matrix((4*n_connections+n_points, 4*n_points))

        # connectivity
        closest_points, is_valid = point_identifier.get_closest_points(source_points)

        if verbose:
            print("N invalid matches: ", is_valid.count(False))

        for i in range(n_connections):
            line = connections[i]
            for j in range(4):
                A[4*i+j, 4*line[0]+j] = -stiff*edge_scale[i]
                A[4*i+j, 4*line[1]+j] = stiff*edge_scale[i]

        # correspondence
        for i in range(n_points):
            if is_valid[i]:
                src_point = list(source_points[i,:])+[1]
            else:
                src_point = [0,0,0,0]

            for j in range(4):
                A[4*n_connections+i, 4*i+j] = src_point[j]

        B = lil_matrix((4*n_connections+n_points, 3)) 
        for i in range(n_points):
            if is_valid[i]:
                tar_point = closest_points[i,:]
            else:
                tar_point = [0,0,0]

            for j in range(3):
                B[4*n_connections + i, j] = tar_point[j]


        # Solve
        A = A.tocsr()
        B = B.tocsr()
        B = A.transpose() * B
        A = A.transpose() * A
        X = spsolve(A,B)

        for i in range(n_points):
            old_point = np.append(source_points[i,:], [1])
            x = X[4*i:4*i+4,:].T
            new_point = x*old_point

            source_points[i,:] = new_point

        if X_old is None:
            delta = float('inf')
        else:
            delta = norm(X-X_old)

        X_old = X
        if verbose:
            print("X delta norm: ", delta)
            print("Loop Duration (s): ", time.time()-start)

        if delta < norm_threshold:
            stiff = stiff / stiffness_step_factor
            if verbose:
                print()
                print("*** Stiffness lowered to: ", stiff)
                print()

    transformed_source_mesh = vtk.vtkPolyData()
    transformed_source_mesh.SetPoints(utils.np_to_vtkPoints(source_points))
    transformed_source_mesh.SetPolys(source_mesh.GetPolys())

    return transformed_source_mesh


def spline(source_mesh, source_landmarks, target_landmarks):
    """
    Applies a thin plate spline transform to the source mesh

    “Principal warps: thin-plate splines and the decomposition of deformations”

    F. L. Bookstein

    :param source_mesh: vtkPolyData to be transformed
    :param source_landmarks: array of shape [n,3]
    :param target_landmarks: array of shape [n,3]
    :return: the transformed source mesh
    """

    src = utils.np_to_vtkPoints(source_landmarks)
    tar = utils.np_to_vtkPoints(target_landmarks)

    spline = vtk.vtkThinPlateSplineTransform()
    spline.SetBasisToR()
    spline.SetTargetLandmarks(tar)
    spline.SetSourceLandmarks(src)
    spline.Update()

    transform = vtk.vtkTransformPolyDataFilter()
    transform.SetTransform(spline)
    transform.SetInputData(source_mesh)
    transform.Update()

    return transform.GetOutput()


def affine_alignment(source_mesh, source_landmarks, target_landmarks, similarity_only = False):
    """
    Applies an affine transformation to the source mesh and landmarks

    :param source_mesh: vtkPolyData to be transformed
    :param source_landmarks: array of shape [n,3]
    :param target_landmarks: array of shape [n,3]

    :return: (transformed source_mesh, transformed source_landmarks)
    """
    tar = utils.np_to_vtkPoints(target_landmarks)
    src = utils.np_to_vtkPoints(source_landmarks)

    src_poly = vtk.vtkPolyData()
    src_poly.SetPoints(src)

    affine = vtk.vtkLandmarkTransform()
    affine.SetSourceLandmarks(src)
    affine.SetTargetLandmarks(tar)
    if similarity_only:
        affine.SetModeToSimilarity()
    else:
        affine.SetModeToAffine()
    affine.Update()

    transform_mesh = vtk.vtkTransformPolyDataFilter()
    transform_mesh.SetTransform(affine)
    transform_mesh.SetInputData(source_mesh)
    transform_mesh.Update()
    transformed_mesh = transform_mesh.GetOutput()

    transform_landmarks = vtk.vtkTransformPolyDataFilter()
    transform_landmarks.SetTransform(affine)
    transform_landmarks.SetInputData(src_poly)
    transform_landmarks.Update()
    transformed_landmarks = utils.vtkPoints_to_np(transform_landmarks.GetOutput().GetPoints())

    return (transformed_mesh, transformed_landmarks)


class PointIdentifier:
    def __init__(self, target_mesh):
        self.point_locator = vtk.vtkPointLocator()
        self.point_locator.SetDataSet(target_mesh)
        self.point_locator.BuildLocator()

        self.cell_locator = vtk.vtkCellLocator()
        self.cell_locator.SetDataSet(target_mesh)
        self.cell_locator.SetTolerance(0.0001)
        self.cell_locator.BuildLocator()

        edge_filter = vtk.vtkFeatureEdges()
        edge_filter.SetInputData(target_mesh)
        edge_filter.BoundaryEdgesOn()
        edge_filter.FeatureEdgesOn()
        edge_filter.SetFeatureAngle(90)
        edge_filter.ManifoldEdgesOff()
        edge_filter.NonManifoldEdgesOff()
        edge_filter.Update()
        edge_points = edge_filter.GetOutput().GetPoints()

        self.edge_point_ids = []
        for i in range(edge_points.GetNumberOfPoints()):
            point_id = self.point_locator.FindClosestPoint(edge_points.GetPoint(i))
            self.edge_point_ids.append(point_id)

    def get_closest_points(self, source_points):
        closest_points = np.zeros(shape=source_points.shape)
        is_valid = []

        for i in range(source_points.shape[0]):
            point = source_points[i,:]

            point_id = self.point_locator.FindClosestPoint(point)
            is_valid.append(not point_id in self.edge_point_ids)

            cellId = vtk.mutable(0)
            closest_point = [0.0, 0.0, 0.0]
            subId = vtk.mutable(0)
            d = vtk.mutable(0.0)
            self.cell_locator.FindClosestPoint(point, closest_point, cellId, subId, d)

            closest_points[i,0] = closest_point[0]
            closest_points[i,1] = closest_point[1]
            closest_points[i,2] = closest_point[2]

        return closest_points, is_valid


class CorrespondenceIdentifier:
    def __init__(self, target_mesh, target_mesh_corresponding_verts, source_mesh_corresponding_verts):
        self.target_mesh_points = utils.vtkPoints_to_np(target_mesh.GetPoints())[target_mesh_corresponding_verts,:]
        self.source_mesh_corresponding_verts = source_mesh_corresponding_verts

    def get_closest_points(self, source_points):
        closest_points = np.zeros(shape=source_points.shape)
        closest_points[self.source_mesh_corresponding_verts,:] = self.target_mesh_points

        is_valid = np.repeat(False, closest_points.shape[0])
        is_valid[self.source_mesh_corresponding_verts] = True

        return closest_points, list(is_valid)
