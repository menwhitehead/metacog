import h5py

def createH5Dataset(filename, data, labels):
    f = h5py.File(filename, 'w')
    f.create_dataset('data', data.shape, data=data, dtype='f4')
    f.create_dataset('labels', labels.shape, data=labels, dtype='f4')
    f.close()

def loadH5Dataset(filename):
    f = h5py.File(filename, 'r')
    X = f['data']
    y = f['labels']
    return X, y
