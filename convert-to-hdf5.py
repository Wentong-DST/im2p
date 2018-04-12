import numpy as np
import h5py
import os
from argparse import ArgumentParser

def save_to_hdf5(args):
    dataset = args.dataset
    # read from disk
    # use float 16 to save space
    boxes = np.fromfile('data_only_im2p_' +dataset+ '_output.h5-boxes', 
                        sep=" ", dtype=np.float16)
    feats = np.fromfile('data_only_im2p_' +dataset+ '_output.h5-feats', 
                        sep=" ", dtype=np.float16)
    print dataset, "raw data shape: ", boxes.shape, feats.shape
    # reshape data
    boxes = np.reshape(boxes, ((boxes.shape[0]/50/4), 50, 4))
    feats = np.reshape(feats, ((feats.shape[0]/50/4096), 50, 4096))
    print dataset, "converted data shape: ", boxes.shape, feats.shape
    
    # write to hdf5
    print 'start writing to h5 file'
    h5file = h5py.File(os.path.join("./data",'im2p_' +dataset+ '_output.h5'), 'w')
    # redundant boxes
    # h5file.create_dataset('boxes', data=boxes)
    h5file.create_dataset('feats', data=feats)
    h5file.close()
    print 'close h5 file'
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='val')
    args = parser.parse_args()
    save_to_hdf5(args)
    
    
