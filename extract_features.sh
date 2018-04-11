
nohup th extract_features.lua -boxes_per_image 50 -max_images -1 -input_txt imgs_train_path.txt -output_h5 ./data/im2p_train_output.h5 -gpu -1 -use_cudnn 0 &

echo 'finish extracting train set'
mv opt.output_h5-feats im2p_train_output.h5-feats
mv opt.output_h5-boxes im2p_train_output.h5-boxes

nohup th extract_features.lua -boxes_per_image 50 -max_images -1 -input_txt imgs_val_path.txt -output_h5 ./data/im2p_val_output.h5 -gpu -1 -use_cudnn 0 &

echo 'finish extracting val set'
mv opt.output_h5-feats im2p_val_output.h5-feats
mv opt.output_h5-boxes im2p_val_output.h5-boxes

nohup th extract_features.lua -boxes_per_image 50 -max_images -1 -input_txt imgs_test_path.txt -output_h5 ./data/im2p_test_output.h5 -gpu -1 -use_cudnn 0 &

echo 'finish extracting test set'
mv opt.output_h5-feats im2p_test_output.h5-feats
mv opt.output_h5-boxes im2p_test_output.h5-boxes