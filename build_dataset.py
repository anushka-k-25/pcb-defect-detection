import os
import shutil
import random

IMG_SIZE = 640

pcb_root = r"C:\Users\anush\pcb-defect-detection\DeepPCB-master\DeepPCB-master\PCBData"
dataset_root = "dataset"

train_img = os.path.join(dataset_root, "images/train")
val_img = os.path.join(dataset_root, "images/val")
train_lbl = os.path.join(dataset_root, "labels/train")
val_lbl = os.path.join(dataset_root, "labels/val")

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = []

for group in os.listdir(pcb_root):

    group_path = os.path.join(pcb_root, group)

    if not os.path.isdir(group_path):
        continue

    folders = os.listdir(group_path)

    img_folder = None
    label_folder = None

    for f in folders:

        if f.endswith("_not"):
            label_folder = os.path.join(group_path, f)
        else:
            img_folder = os.path.join(group_path, f)

    if img_folder is None or label_folder is None:
        continue

    for file in os.listdir(img_folder):

        if file.endswith("_test.jpg"):
            images.append((img_folder, label_folder, file))

random.shuffle(images)

split = int(len(images) * 0.8)

train_set = images[:split]
val_set = images[split:]

def process(dataset, img_dst, lbl_dst):

    for img_folder, label_folder, img_file in dataset:

        name = img_file.replace("_test.jpg", "")

        src_img = os.path.join(img_folder, img_file)
        dst_img = os.path.join(img_dst, img_file)

        shutil.copy(src_img, dst_img)

        label_file = name + ".txt"
        src_lbl = os.path.join(label_folder, label_file)

        if not os.path.exists(src_lbl):
            continue

        dst_lbl = os.path.join(lbl_dst, name + "_test.txt")

        new_lines = []

        with open(src_lbl, "r") as f:
            lines = f.readlines()

        for line in lines:

            x1, y1, x2, y2, cls = map(float, line.split())

            xc = (x1 + x2) / 2 / IMG_SIZE
            yc = (y1 + y2) / 2 / IMG_SIZE
            w = (x2 - x1) / IMG_SIZE
            h = (y2 - y1) / IMG_SIZE

            new_lines.append(f"{int(cls)} {xc} {yc} {w} {h}\n")

        with open(dst_lbl, "w") as f:
            f.writelines(new_lines)


process(train_set, train_img, train_lbl)
process(val_set, val_img, val_lbl)

print("Dataset build complete")