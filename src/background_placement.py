import math
import numpy as np
from PIL import Image, ImageOps
import os
import imageio as io
import pickle

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_and_crop(image, angle):
    image_height, image_width = image.shape[0:2]
    return crop_around_center(
            np.array(Image.fromarray(image).rotate(angle,)),
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(angle)
            )
        )


def get_random_background(w, h, bg_path):
    bgs = os.listdir(bg_path)
    bg = io.imread(bg_path + bgs[np.random.randint(len(bgs))])
    bg = rotate_and_crop(bg, np.random.randint(360))
    x, y = np.random.randint(max((bg.shape[1]-w, 1))), np.random.randint(max(bg.shape[0]-h, 1))
    bg = bg[y:y+h, x:x+w, :]
    if np.all(bg.shape[:2] == np.array([h, w])):
        return bg
    return get_random_background(w, h, bg_path) #didn't work, try again


def scale_with_bbox(img, bboxes):
    assert img.shape[0] == img.shape[1] # assuming square
    img_size = img.shape[0]
    new_size = np.random.randint(img_size * 0.5, img_size)
    scale = new_size / img_size

    bbarr = np.array(bboxes[1:], dtype=float).reshape(-1,5)

    if type(bboxes[1]) == np.int64:
        # convert to floats
        bbarr[:,:4] = bbarr[:,:4]/img_size

    img = Image.fromarray(img)
    img = ImageOps.scale(img, scale)

    if np.random.random() > 0.5: # flip vert
        img = ImageOps.flip(img)
        bbarr[:,1::2] = 1 - bbarr[:,1::2]
        bbarr[:,[1,3]] = bbarr[:,[3,1]]
    
    if np.random.random() > 0.5: # flip horiz
        img = ImageOps.mirror(img)
        bbarr[:,0::2] = 1 - bbarr[:,0::2]
        bbarr[:,[0,2]] = bbarr[:,[2,0]]

    for i in range(4):
        bboxes[i+1::5] = bbarr[:,i]

    return np.array(img), bboxes


def place_on_bg(img, bg, bboxes):
    background = Image.fromarray(bg)
    foreground = Image.fromarray(img)

    offset = np.random.randint(bg.shape[0] - img.shape[0], size=2)
    
    background.paste(foreground, tuple(offset), foreground)

    bbarr = np.array(bboxes[1:], dtype=float).reshape(-1,5)
    bbarr = bbarr * img.shape[0] / bg.shape[0]
    bbarr[:,:2] += offset / bg.shape[0]
    bbarr[:,2:4] += offset / bg.shape[0]

    for i in range(4):
        bboxes[i+1::5] = bbarr[:,i]
        
    return np.array(background), bboxes


def make_combined_images(transparent_path, bg_path, output_path):
    metadata = pickle.load(open(transparent_path + 'metadata.pkl', 'rb'))
    classes = ['LEGS', 'TORSO', 'HEAD', 'HATS', 'MINIFIG']

    metadata.drop(metadata.loc[metadata['bboxes'].apply(lambda a: min(a[1:])) < 0].index, inplace=True) #drop faulty bboxes
    metadata = metadata.sample(frac=1).reset_index(drop=True) #shuffle and reindex

    anno_lines = []
    for i, row in metadata.iterrows():
        foreground = io.imread(transparent_path + row['filename'])
        bboxes = row['bboxes']

        foreground, bboxes = scale_with_bbox(foreground, bboxes)
        combined, bboxes = place_on_bg(foreground, get_random_background(500, 500, bg_path), bboxes)

        anno_lines.append(', '.join([str(x) for x in bboxes]) + '\n')

        io.imwrite(output_path + row['filename'], combined)

        #bboxes = np.array(bboxes[1:]).reshape(-1,5)
        #img_box = draw_bbox(foreground[...,:3].copy(), bboxes, classes=classes, show_label=True, probability=False)
    #print(anno_lines)
    with open(output_path + 'annotations.txt', 'w') as file:
        file.writelines(anno_lines)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import draw_bbox

    project_path = '/home/darrel/Documents/gal/projects/leggo_my_legs/datagen/minifig_rendered_try2/'
    images_path = project_path + 'images/'
    pov_output_path = images_path + 'transparent/'
    combined_output_path = images_path + 'combined/'
    bg_full_path = images_path + 'backgrounds/'

    make_combined_images(pov_output_path, bg_full_path, combined_output_path)