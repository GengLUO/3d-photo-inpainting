import os
import cv2
import glob
import numpy as np
import imageio
import shutil

DEPTH_ANYTHING_BASE = 'Depth-Anything'

DEPTH_ANYTHING_INPUTS = 'inputs'
DEPTH_ANYTHING_OUTPUTS = 'outputs'

def run_depth_anything(img_names, src_folder, depth_folder):

    if not isinstance(img_names, list):
        img_names = [img_names]

    # Ensure base, input, and output directories exist
    input_path = os.path.join(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_INPUTS)
    output_path = os.path.join(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_OUTPUTS)

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Clean the folders to remove irrelevant files
    clean_folder(input_path)
    clean_folder(output_path)

    tgt_names = []
    for img_name in img_names:
        base_name = os.path.basename(img_name)
        tgt_name = os.path.join(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_INPUTS, base_name)
        os.system(f'cp {img_name} {tgt_name}')

        # keep only the file name here.
        # they save all depth as .png file
        tgt_names.append(os.path.basename(tgt_name).replace('.jpg', '.png'))

    # os.system(f'cd {DEPTH_ANYTHING_BASE} && python run.py --Final --data_dir {DEPTH_ANYTHING_INPUTS}/  --output_dir {DEPTH_ANYTHING_OUTPUTS} --depthNet 0')
    os.system(f'cd {DEPTH_ANYTHING_BASE} && python run.py --encoder vits --img-path {DEPTH_ANYTHING_INPUTS}/ --outdir {DEPTH_ANYTHING_OUTPUTS} --pred-only --grayscale')

    # Copy the results from the outputs folder to the depth_folder
    for file_name in os.listdir(output_path):
        src_file = os.path.join(output_path, file_name)
        dst_file = os.path.join(depth_folder, file_name)
        shutil.copy(src_file, dst_file)

def clean_folder(folder, img_exts=['.png', '.jpg', '.npy']):

    for img_ext in img_exts:
        paths_to_check = os.path.join(folder, f'*{img_ext}')
        if len(glob.glob(paths_to_check)) == 0:
            continue
        print(paths_to_check)
        os.system(f'rm {paths_to_check}')

def resize_depth(depth, width, height):
    """Resize numpy (or image read by imageio) depth map

    Args:
        depth (numpy): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = cv2.blur(depth, (3, 3))
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
