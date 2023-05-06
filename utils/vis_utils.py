import matplotlib.pyplot as plt
import numpy as np
import os
from utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data

PANDAS_ID_TO_BGR = {
    0 : [10, 90, 0],
    1 : [200, 0, 10],
    2 : [0, 10, 255],
    3: [245, 150, 100],
    4: [245, 230, 100],
    5: [250, 80, 100],
    6: [150, 60, 30],
    7: [255, 0, 0],
    8: [180, 30, 80],
    9: [255, 0, 0],
    10: [30, 30, 255],
    11: [200, 40, 255],
    12: [90, 30, 150],
    13: [255, 0, 255],
    14: [255, 150, 255],
    15: [75, 0, 75],
    16: [75, 0, 175],
    17: [0, 200, 255],
    18: [50, 120, 255],
    19: [0, 150, 255],
    20: [170, 255, 150],
    21: [0, 175, 0],
    22: [0, 60, 135],
    23: [80, 240, 150],
    24: [150, 240, 255],
    25: [0, 0, 255],
    26: [255, 255, 50],
    27: [245, 150, 100],
    28: [255, 0, 0],
    29: [200, 40, 255],
    30: [30, 30, 255],
    31: [90, 30, 150],
    32: [250, 80, 100],
    33: [180, 30, 80],
    34: [255, 0, 0],
    35: [155, 255, 50],
    36: [45, 150, 100],
    37: [25, 0, 0],
    38: [20, 40, 255],
    39: [130, 30, 255],
    40: [190, 30, 150],
    41: [25, 80, 100],
    42: [18, 30, 80]
}

PANDAS_COLOR_PALETTE = [PANDAS_ID_TO_BGR[id] if id in PANDAS_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(PANDAS_ID_TO_BGR.keys())[-1] + 1)]

def write_obj(points, file, rgb=False):
    fout = open('%s.obj' % file, 'w')
    for i in range(points.shape[0]):
        if not rgb:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], 255, 255, 0))
        else:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], points[i, -3] * 255, points[i, -2] * 255,
                points[i, -1] * 255))


def draw_points_image_labels(img, img_indices, seg_labels, show=True, color_palette_type='Pandas', point_size=3.5):
    if color_palette_type == 'Pandas':
        color_palette = PANDAS_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')

    color_palette = np.array(color_palette) / 255.
    # seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels[:, 0]]
    colors = colors[:, [2, 1, 0]]
    plt.figure(figsize=(10, 6))
    #plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        os.makedirs('visualization', exist_ok=True)
        #plt.show()
        plt.savefig('./visualization/points.png')
        plt.savefig('./visualization/image.png')


def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)


def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    colors = []
    for depth_val in depth:
        colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    # ax5.imshow(np.full_like(img, 255))
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        os.makedirs('visualization', exist_ok=True)
        #plt.show()
        plt.savefig('./visualization/depth.png')


def draw_bird_eye_view(coords, full_scale=4096):
    os.makedirs('visualization', exist_ok=True)
    plt.scatter(coords[:, 0], coords[:, 1], s=0.1)
    plt.xlim([0, full_scale])
    plt.ylim([0, full_scale])
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    plt.savefig('./visualization/bird_eye.png')
