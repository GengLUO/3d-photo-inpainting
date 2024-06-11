import os
import shutil
from pathlib import Path
import subprocess

DEPTH_ANYTHING_BASE = 'Depth-Anything'
DEPTH_ANYTHING_INPUTS = 'inputs'
DEPTH_ANYTHING_OUTPUTS = 'outputs'

def run_depth_anything(img_name, src_folder, depth_folder):
    input_path = Path(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_INPUTS)
    output_path = Path(DEPTH_ANYTHING_BASE, DEPTH_ANYTHING_OUTPUTS)
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_folder(input_path)
    clean_folder(output_path)

    base_name = Path(img_name).name
    tgt_name = input_path / base_name
    shutil.copy(img_name, tgt_name)

    command = [
        'python', 'run.py',
        '--encoder', 'vits',
        '--img-path', str(input_path) + '/',
        '--outdir', str(output_path),
        '--pred-only', '--grayscale'
    ]

    subprocess.run(command, cwd=str(Path(DEPTH_ANYTHING_BASE)), check=True)

    for file_name in output_path.iterdir():
        shutil.copy(file_name, Path(depth_folder, file_name.name))

def clean_folder(folder_path):
    for ext in ['.png', '.jpg', '.npy']:
        for file in folder_path.glob(f'*{ext}'):
            file.unlink()
