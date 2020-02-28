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
    """
    returns a dictionary of lines to be modified.
    Keys are the beginning of the lines and values are the new endings
    """
    return {
        '#declare LDXCameraLoc =': f' < {", ".join(camera_vector.astype(str))} >;',
        '#declare LDXCameraSky =': ' <0,-1,0>;',
        '#declare LDXFloor = ': '0;',
        # '#declare LDXFloorLoc =': ' LDXMaxY + 10;',
        '#declare LDXCameraLookAt =': ' LDXCenter;',
        '	angle ': '40',
    }

def add_lines_for_background(bg):
    """
    Depricated, using alpha channel instead
    """
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

def extract_data(lines):
    """
    x_items is a dictionary where keys are the beginnings of a lines
    that we'd like to extract data from and values are the keys in
    the returned dictionary of extracted values.
    """

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

    data = {}
    part_list = None
    for i, line in enumerate(lines):
        if line == '// Unofficial Model\n':
            part_list = []
            data['part_list_start'] = i
        if part_list is not None and line == '	object {\n':
            part_list.append(i+1)

        val = None
        split = line.split(' = ')
        if split[0] in x_items:
            val = float(split[1][:-2])
            
        else:
            split = line.split(' ')
            if split[0] in x_items:
                val = float(split[1])
                
        if val is not None:
            key = x_items[split[0]]
            if key in data:
                if type(data[key]) == list:
                    data[key].append(val)
                else:
                    data[key] = [data[key], val]
            else:
                data[key] = val

    part_list = [part_name(lines[i]) for i in part_list]
    part_list = [p for p in part_list if p != ""]
    data['parts'] = part_list

    return data

def part_name(s):
    s = s.strip()
    if s.endswith('_dot_dat'):
        s = s[:-8]
    if s.endswith('_clear'):
        s = s[:-6]
    if s.startswith('LDX_'):
        s = s[4:]
    if s.startswith('lg_'):
        s = s[3:]
    s = s.lstrip('0') #remove leading zeros
    return s


def replace(line, line_modifications):
    """
    takes a list of lines and a dictionary of replacements to make
    and performs the replacements.
    Returns list of modified lines.
    """
    for k, v in line_modifications.items():
        if line[:len(k)] == k:
            return k + v + '\n'
    return line

def randomize_pov_files(path, ini_file):
    """
    finds all .pov files in a directory and modifies them
    """
    categories = get_part_categories(ini_file, True)
    meta = pd.DataFrame(columns=['filename', 'camera_vector', ])

    line_modifications, camera_vector = None, None
    pov_files = os.listdir(path)
    pov_files = [file for file in pov_files if file[-4:] == '.pov']
    for filename in pov_files:
        camera_vector = random_three_vector()
        line_modifications = modify_lines(camera_vector)
        with open(f'{path}{filename}', 'r+') as file:
            lines = file.readlines()
            x_data = extract_data(lines)
            new_lines = list(map(lambda l: replace(l, line_modifications), lines))
            new_lines = comment_parts(new_lines, categories, x_data, ['HEAD', 'LEGS'])
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
    
    meta['part_list_start'] = meta['part_list_start'].astype(int)
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


def find_next(lines, i_start, target):
    for i in range(i_start, len(lines)):
        if lines[i].rstrip('\n').lstrip('/') == target:
            return i
    return len(lines) + 1

def set_comment(lines, start, end, comment):
    for i in range(start, end + 1):
        lines[i] = ('//' if comment else '') + lines[i].lstrip('/')

        
def comment_parts(lines, categories, data, categories_allowed):
    i = data['part_list_start']
    i_stop = find_next(lines, i, '}')
    while True:
        i = find_next(lines, i+1, '	object {')
        i_end = find_next(lines, i+1, '	}')
        if i_end >= i_stop:
            break
        part = part_name(lines[i + 1].lstrip('/'))
        set_comment(lines, i, i_end, not (categories.loc[part, "Category"] in categories_allowed))
    return lines


def get_part_categories(ini_filename, abridged=False):
    parts = pd.DataFrame(columns=['Name', "Number", "SubCategory"])

    with open(ini_filename) as f:
        ini_file = f.readlines()

    ini_file = [l for l in ini_file if not (l.startswith(';') or l.startswith('"----') or l == '\n')]

    current_cat = None
    for l in ini_file:
        if l.startswith('['):
            current_cat = l.strip('[]\n ')
        else:
            split = l.split('"')
            if len(split) >= 4 and split[3] != "":
                parts = parts.append(pd.DataFrame(data={"SubCategory": [current_cat], "Name": [split[1]], "Number": [split[3][:-4].lower()]}), ignore_index=True)

    parts['Category'] = parts['SubCategory'].map({
        'HATS': 'HATS',
        'HATS2': 'HATS',
        'HEAD': 'HEAD',
        'BODY': 'TORSO',
        'BODY2': 'LEGS',
        'BODY3': 'LEGS',
        'NECK': 'TORSO',
        'LARM': 'TORSO',
        'RARM': 'TORSO',
        'LHAND': 'TORSO',
        'RHAND': 'TORSO',
        'LHANDA': 'TORSO',
        'RHANDA': 'TORSO',
        'LLEG': 'LEGS',
        'RLEG': 'LEGS',
        'LLEGA': 'LEGS',
        'RLEGA': 'LEGS'
    })

    # this one part is in two categories
    parts.loc[parts['Number'] == "61190c", 'Category'] = 'LEGS'
    # pulling one non-part line
    parts.drop(parts[parts['Number'] == " on transparent backgr"].index, inplace=True)

    if abridged:
        parts = parts[['Number', 'Category']].groupby(['Number', 'Category']).first()
        return parts.reset_index().set_index('Number')

    return parts


if __name__ == "__main__":
    project_path = '/home/darrel/Documents/gal/projects/leggo_my_legs/datagen/minifig_rendered_try2/'
    models_path = project_path + 'models_full/'
    pov_path = project_path + 'pov_models/'
    images_path = project_path + 'images/'
    pov_output_path = images_path + 'transparent/'
    combined_output_path = images_path + 'combined/'
    povray_path = 'povray'
    bg_full_path = images_path + 'backgrounds/'
    bg_random_path = images_path + 'backgrounds_random/'

    metadata = randomize_pov_files(pov_path, '/home/darrel/Documents/gal/projects/leggo_my_legs/datagen/MLCad.ini')

    print('Done')
    print('asdlfj')