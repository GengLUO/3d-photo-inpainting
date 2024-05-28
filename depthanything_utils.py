import os
import cv2
import glob
import numpy as np
import imageio

DEPTH_ANYTHING_BASE = 'Depth-Anything'

DEPTH_ANYTHING_INPUTS = 'inputs'
DEPTH_ANYTHING_OUTPUTS = 'outputs'

def run_depth_anything(img_names, src_folder, depth_folder):

    if not isinstance(img_names, list):
        img_names = [img_names]

    # remove irrelevant files first
    clean_folder(os.path.join(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_INPUTS))
    clean_folder(os.path.join(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_OUTPUTS))

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
