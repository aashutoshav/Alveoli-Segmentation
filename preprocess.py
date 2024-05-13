import os
import cv2

image_path = "./images"
mask_path = "./masks"

for filename in os.listdir(image_path):
    old_filename = os.path.join(image_path, filename)
    new_filename = filename.split("_")[-1].split("_XY")[0][3:]
    os.rename(old_filename, os.path.join(image_path, new_filename))

for filename in os.listdir(mask_path):
    old_filename = os.path.join(mask_path, filename)
    new_filename = filename.split("_")[-1].split("_XY")[0][3:]
    os.rename(old_filename, os.path.join(mask_path, new_filename))

target_size = (256, 256)

for idx, (img_filename, mask_filename) in enumerate(
    zip(os.listdir(image_path), os.listdir(mask_path))
):
    img = cv2.imread(os.path.join(image_path, img_filename))
    if img.shape != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    mask = cv2.imread(os.path.join(mask_path, mask_filename))
    if mask.shape != target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)

    print(f"Image {idx+1} shape: {img.shape} | Mask {idx+1} shape: {mask.shape}")
