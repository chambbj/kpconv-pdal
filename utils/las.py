import json
import numpy as np
import pdal

def read_las_points(filename):
    p = pdal.Pipeline(json.dumps([filename]))
    p.validate()
    p.execute()
    data = p.arrays[0]
    points = np.vstack((data['X'], data['Y'], data['Z'])).T
    return points

def read_processed_las(filename):
    p = pdal.Pipeline(json.dumps([filename]))
    p.validate()
    p.execute()
    data = p.arrays[0]
    points = np.vstack((data['X'], data['Y'], data['Z'])).T
    # intensity = np.expand_dims(data['Intensity'], 1).astype(np.float32)
    # intensity = np.minimum(intensity, 255.0)/255.0
    # features = intensity
    # rn = data['ReturnNumber'].astype(np.float32)
    # nr = data['NumberOfReturns'].astype(np.float32)
    features = np.vstack((data['Linearity'],data['Planarity'],data['Scattering'],data['Verticality'])).T
    # features = np.vstack((data['Eigenvalue0'],data['Eigenvalue1'],data['Eigenvalue2'])).T
    # features = np.vstack((data['X'], data['Y'], data['Z'])).T
    labels = data['Classification']
    return points, features, labels

def read_raw_las(filename):
    p = pdal.Pipeline(json.dumps([
        # filename
        filename,
        {
            "type":"filters.range",
            "limits":"Classification(:17]"
        },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[1:1]=0"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[2:2]=1"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[7:7]=2"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[9:9]=3"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[17:17]=4"
        # },
        {
            "type":"filters.covariancefeatures"
        }
        # {
        #     "type":"filters.eigenvalues",
        #     "knn":10
        # }
    ]))
    p.validate()
    p.execute()
    data = p.arrays[0]
    points = np.vstack((data['X'], data['Y'], data['Z'])).T
    # intensity = np.expand_dims(data['Intensity'], 1).astype(np.float32)
    # intensity = np.minimum(intensity, 255.0)/255.0
    # features = intensity
    # rn = data['ReturnNumber'].astype(np.float32)
    # nr = data['NumberOfReturns'].astype(np.float32)
    features = np.vstack((data['Linearity'],data['Planarity'],data['Scattering'],data['Verticality'])).T
    # features = np.vstack((data['Eigenvalue0'],data['Eigenvalue1'],data['Eigenvalue2'])).T
    # features = np.vstack((data['X'], data['Y'], data['Z'])).T
    labels = data['Classification']
    return points, features, labels

def read_subsampled_las(filename, dl):
    p = pdal.Pipeline(json.dumps([
        # filename
        filename,
        {
            "type":"filters.range",
            "limits":"Classification(:17]"
        },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[1:1]=0"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[2:2]=1"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[7:7]=2"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[9:9]=3"
        # },
        # {
        #     "type":"filters.assign",
        #     "assignment":"Classification[17:17]=4"
        # },
        {
            "type":"filters.covariancefeatures"
        },
        {
            "type":"filters.sample",
            "radius":dl
        }
        # {
        #     "type":"filters.eigenvalues",
        #     "knn":10
        # }
    ]))
    p.validate()
    p.execute()
    data = p.arrays[0]
    points = np.vstack((data['X'], data['Y'], data['Z'])).T
    # intensity = np.expand_dims(data['Intensity'], 1).astype(np.float32)
    # intensity = np.minimum(intensity, 255.0)/255.0
    # features = intensity
    # rn = data['ReturnNumber'].astype(np.float32)
    # nr = data['NumberOfReturns'].astype(np.float32)
    features = np.vstack((data['Linearity'],data['Planarity'],data['Scattering'],data['Verticality'])).T
    # features = np.vstack((data['Eigenvalue0'],data['Eigenvalue1'],data['Eigenvalue2'])).T
    # features = np.vstack((data['X'], data['Y'], data['Z'])).T
    labels = data['Classification']
    return points, features, labels

def write_las(filename, array):
    # merge the fields then
    p = pdal.Pipeline(json.dumps([{
        "type":"writers.las",
        "filename":filename,
        # "offset_x":"auto",
        # "offset_y":"auto",
        # "offset_z":"auto",
        # "scale_x":0.01,
        # "scale_y":0.01,
        # "scale_z":0.01
        "forward":"all",
        "minor_version":4,
        "extra_dims":"all"
        }]), [array])
    p.validate()
    p.execute()
    return True