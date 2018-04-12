# some machine may not dump all data onto disk correctly
# if so, try another machine

sed '1,17d' im2p_val_output.h5-boxes > data_only_im2p_val_output.h5-boxes
sed '1,17d' im2p_test_output.h5-boxes > data_only_im2p_test_output.h5-boxes
sed '1,17d' im2p_train_output.h5-boxes > data_only_im2p_train_output.h5-boxes
sed '1,17d' im2p_val_output.h5-feats > data_only_im2p_val_output.h5-feats
sed '1,17d' im2p_test_output.h5-feats > data_only_im2p_test_output.h5-feats
sed '1,17d' im2p_train_output.h5-feats > data_only_im2p_train_output.h5-feats


python convert-to-hdf5.py --dataset 'val'
python convert-to-hdf5.py --dataset 'test'
python convert-to-hdf5.py --dataset 'train'

