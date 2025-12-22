import os, shutil

root = "./data/tiny-imagenet"
val_dir = os.path.join(root, "val")
images_dir = os.path.join(val_dir, "images")
ann_file = os.path.join(val_dir, "val_annotations.txt")

with open(ann_file) as f:
    for line in f:
        fname, wnid = line.strip().split('\t')[:2]
        cls_dir = os.path.join(val_dir, wnid)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.move(os.path.join(images_dir, fname), os.path.join(cls_dir, fname))
shutil.rmtree(images_dir)
print("✅ Validation split prepared.")
