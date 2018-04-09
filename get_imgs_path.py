#! encoding: UTF-8

import os
import glob

# ------------------------------------------------------------------------------------------------------
# Training Images
# ------------------------------------------------------------------------------------------------------

train_images_path = "./data/genome/im2p_train"
train_images = glob.glob(train_images_path + "/*.jpg")

f = open("imgs_train_path.txt", "w")
for item in train_images:
    f.write(item + "\n")
f.close()

# ------------------------------------------------------------------------------------------------------
# Validation Images
# ------------------------------------------------------------------------------------------------------

train_images_path = "./data/genome/im2p_val"
train_images = glob.glob(train_images_path + "/*.jpg")

f = open("imgs_val_path.txt", "w")
for item in train_images:
    f.write(item + "\n")
f.close()

# ------------------------------------------------------------------------------------------------------
# Test Images
# ------------------------------------------------------------------------------------------------------

train_images_path = "./data/genome/im2p_test"
train_images = glob.glob(train_images_path + "/*.jpg")

f = open("imgs_test_path.txt", "w")
for item in train_images:
    f.write(item + "\n")
f.close()