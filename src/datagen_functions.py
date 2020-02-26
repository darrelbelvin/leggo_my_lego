"""
A collection of functions used to create a dataset of lego minifigures using MCad, LDView, and POVRay
"""
import os
import subprocess
import numpy as np
import pandas as pd

def random_three_vector(v_scale=300):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    _x = np.sin(theta) * np.cos(phi)
    _y = np.sin(theta) * np.sin(phi)
    _z = np.cos(theta)
    return np.array((_x, _y, _z)) * v_scale

def modify_lines(camera_vector):
    return {
        '#declare LDXCameraLoc =': f' < {", ".join(camera_vector.astype(str))} >;',
        '#declare LDXCameraSky =': ' <0,-1,0>;',
        '#declare LDXFloor = ': '0;',
        # '#declare LDXFloorLoc =': ' LDXMaxY + 10;',
        '#declare LDXCameraLookAt =': ' LDXCenter;',
        '	angle ': '40',
    }

def add_lines(bg):
    return [
        '\n', '\n',
        '#include "screen.inc"\n',
        '\n',
        'Set_Camera(LDXCameraLoc, LDXCameraLookAt, 70) \n',
        'Set_Camera_Sky(LDXCameraSky)\n',
        '\n',
        '#declare MyScreenTexture =\n',
        'texture {\n',
        f'   pigment {{ image_map {{ png "{bg}" interpolate 2 }} }}\n',
        "   finish {ambient 1 diffuse 0} // Don't want any shadows on it\n",
        '   scale 0.5 translate <0.5,0.5,0> // move it into the <0,0><1,1> region\n',
        '}\n',
        '\n',
        'Screen_Plane ( MyScreenTexture, 300, <-1,-1>, <1,1> )'
    ]

x_items = {
    '#declare LDXMinX': 'MinX',
    '#declare LDXMinY': 'MinY',
    '#declare LDXMinZ': 'MinZ',
    '#declare LDXMaxX': 'MaxX',
    '#declare LDXMaxY': 'MaxY',
    '#declare LDXMaxZ': 'MaxZ',
    '#declare LDXCenterX': 'CenterX',
    '#declare LDXCenterY': 'CenterY',
    '#declare LDXCenterZ': 'CenterZ',
    '	angle': 'camera_angle',
}

def extract_data(lines):
    data = {}
    for line in lines:
        split = line.split(' = ')
        if split[0] in x_items:
            data[x_items[split[0]]] = float(split[1][:-2])
        else:
            split = line.split(' ')
            if split[0] in x_items:
                data[x_items[split[0]]] = float(split[1])
    return data

def replace(line, vector):
    for k, v in modify_lines(vector).items():
        if line[:len(k)] == k:
            return k + v + '\n'
    return line

def randomize_pov_files(path):
    meta = pd.DataFrame(columns=['filename', 'camera_vector', ])

    camera_vector = None
    pov_files = os.listdir(path)
    pov_files = [file for file in pov_files if file[-4:] == '.pov']
    for filename in pov_files:
        camera_vector = random_three_vector()
        with open(f'{path}{filename}', 'r+') as file:
            lines = file.readlines()
            x_data = extract_data(lines)
            new_lines = list(map(lambda l: replace(l, camera_vector), lines))
            #these_add_lines = add_lines("1 - Copy.png")
            #if new_lines[-1] != these_add_lines[-1]:
            #    new_lines.append('\n')
            #    new_lines.extend(these_add_lines)
            #else:
            #    new_lines[-len(these_add_lines):] = these_add_lines
            file.seek(0)
            file.truncate()
            file.writelines(new_lines)
        x_data.update({'filename' : filename, 'camera_vector' : camera_vector})
        meta = meta.append(x_data, ignore_index=True)
    return meta

def render_pov_files(path):
    pov_files = os.listdir(path)
    pov_files = [file for file in pov_files if file[-4:] == '.pov']
    for filename in pov_files:
        #cmd = f'{povray_path} "{path}povray.ini" /RENDER "{path}{filename}" /NR /EXIT'
        cmd = f'povray "{path}povray.ini" "{path}{filename}"'
        #print(cmd)
        subprocess.call(cmd, shell=True)
        #!$cmd
