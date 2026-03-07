import os
import shutil

dataset_root = r"DeepPCB-master/DeepPCB-master/PCBData"

train_file = os.path.join(dataset_root, "trainval.txt")
test_file = os.path.join(dataset_root, "test.txt")

train_img = "dataset/images/train"
val_img = "dataset/images/val"

train_lbl = "dataset/labels/train"
val_lbl = "dataset/labels/val"

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)


def process(list_file, img_dest, lbl_dest):

    with open(list_file) as f:
        lines = f.readlines()

    for line in lines:

        img_rel, lbl_rel = line.strip().split()

        # change filename to *_test.jpg
        base = os.path.basename(img_rel).replace(".jpg", "_test.jpg")

        folder = os.path.dirname(img_rel)

        img_path = os.path.join(dataset_root, folder, base)
        lbl_path = os.path.join(dataset_root, lbl_rel)

        if os.path.exists(img_path):
            shutil.copy(img_path, img_dest)

        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, lbl_dest)


print("Preparing training set...")
process(train_file, train_img, train_lbl)

print("Preparing validation set...")
process(test_file, val_img, val_lbl)

print("Dataset ready!")