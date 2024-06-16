import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from voxel_gan import Generator
import os
import torch

if __name__ == '__main__':
    mode = input('Visualize existing data (y / n): ')
    if mode == 'y':
        con = sqlite3.connect('voxel.db')
        with con:
            cur = con.cursor()
            build_ids = cur.execute('select id from build limit 10').fetchall()
            for build_id in build_ids:
                build_id = build_id[0]
                mats = cur.execute('select * from material').fetchall()
                mats = {m[0] : m[1] for m in mats}
                voxels = cur.execute('select * from block where build_id = \'' + build_id + '\'').fetchall()
                minx, maxx, miny, maxy, minz, maxz = float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')
                for v in voxels:
                    minx = min(v[2], minx)
                    maxx = max(v[2], maxx)
                    miny = min(v[3], miny)
                    maxy = max(v[3], maxy)
                    minz = min(v[4], minz)
                    maxz = max(v[4], maxz)
                sizex, sizey, sizez = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
                n_voxels = np.zeros((sizex, sizey, sizez), dtype = bool)
                facecolors = np.full((sizex, sizey, sizez), '#00000000', dtype = 'U9')
                edgecolors = np.full((sizex, sizey, sizez), '#00000000', dtype = 'U9')
                voxel_set = set()
                for v in voxels:
                    x, y, z = v[2] - minx, v[3] - miny, v[4] - minz
                    voxel_set.add((x, y, z))
                for v in voxels:
                    x, y, z = v[2] - minx, v[3] - miny, v[4] - minz
                    if len(voxel_set & {(x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)}) == 6:
                        continue
                    mat_id = v[1]
                    n_voxels[x, y, z] = True
                    facecolors[x, y, z] = mats[mat_id].ljust(7, '0') + 'FF'
                ax = plt.figure().add_subplot(projection = '3d')
                ax.voxels(n_voxels, facecolors = facecolors, edgecolors = edgecolors)
                plt.show()
    else:
        models_path = 'models/'
        epoch = input('Epoch count: ')
        path = models_path + 'voxel_' + epoch + '.pt'
        if not os.path.exists(path):
            print('Model not found')
            exit()
        noise_size = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gen = Generator(noise_size).to(device)
        checkpoint = torch.load(path)
        gen.load_state_dict(checkpoint['generator_state_dict'])
        noise = torch.randn(10, noise_size, 1, 1, 1).to(device)
        fake_images = gen(noise)
        for image in fake_images:
            n_voxels = np.ones((32, 32, 32), dtype = bool)
            facecolors = np.full((32, 32, 32), '#00000000', dtype = 'U9')
            edgecolors = np.full((32, 32, 32), '#00000000', dtype = 'U9')
            for x in range(32):
                for y in range(32):
                    for z in range(32):
                        rgba = tuple(map(int, torch.round(((image[:, x, y, z] + 1) / 2) * 255)))
                        if rgba[-1] == 0:
                            n_voxels[x, y, z] = False
                        hex_color = '#{:02x}{:02x}{:02x}{:02x}'.format(*rgba)
                        facecolors[x, y, z] = hex_color
            ax = plt.figure().add_subplot(projection = '3d')
            ax.voxels(n_voxels, facecolors = facecolors, edgecolors = edgecolors)
            plt.show()