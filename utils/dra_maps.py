# -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import pdb
# from lightdirection import *

# matplotlib.use('TkAgg')

class Tsai(object):

    def __init__(self,  input_fname='', local_estimation=False, light_direction='constant'):

        self.input_fname = input_fname

        self.depth_map = None
        self.albedo_map = None
        self.reflectance_map = None

        self.normals = None
        self.albedo_free_image = None
        self.albedo_free_depth = None

        self.NB_ITERATIONS = 5
        self.Wn = 0.001

        self.local_estimation = local_estimation
        self.light_direction = light_direction

        self.debug = False

    def load_image(self):
        #self.image = np.asarray(cv2.imread(self.input_fname, cv2.IMREAD_GRAYSCALE))
        self.image = self.input_fname
    @staticmethod
    def normalization(data, min_value=0., max_value=255.):
        data = np.array(data, dtype=np.float32)

        data_min = data.min()
        data_max = data.max()

        if data_min != data_max:
            data_norm = (data - data_min) / (data_max - data_min)
            data_scaled = data_norm * (max_value - min_value) + min_value
        else:
            data_scaled = data

        return data_scaled

    # @profile  # -- used for line_profile and memory_profiler packages
    def __compute_global_sfs(self):

        self.image = np.array(self.image, dtype=np.float32)

        image_shape = self.image.shape[:2]

        self.depth_map = np.zeros(image_shape, dtype=np.float32)
        self.reflectance_map = np.zeros(image_shape, dtype=np.float32)
        self.albedo_map = np.zeros(image_shape, dtype=np.float32)

        zk1 = np.zeros(image_shape, dtype=np.float32)
        sk1 = np.ones(image_shape, dtype=np.float32)

        if 'constant' in self.light_direction:
            light = [0.1, 0.1, 1.0]
            xl, yl, zl = light
        else:
            raise(Exception, 'Method for light estimation not found!')

        try:
            max_pixel = self.image.max()
            assert max_pixel != 0, 'invalid value for max value pixel!'
        except AssertionError:
            # -- the image is flat so there is nothing to do
            return None

        ps = np.ones(image_shape, dtype=np.float32) * (xl / zl)  # 0.0
        qs = np.ones(image_shape, dtype=np.float32) * (yl / zl)  # 1.0

        for it in range(self.NB_ITERATIONS):

            # -- compute gradient
            p, q = np.gradient(zk1)

            # -- create normal vector for each surface point
            self.normals = np.dstack([-p, -q, np.ones(p.shape)])

            # -- compute the norm of each surface point
            norm = np.linalg.norm(self.normals, axis=2)

            # -- compute the unit vectors
            self.normals[:, :, 0] /= norm
            self.normals[:, :, 1] /= norm
            self.normals[:, :, 2] /= norm

            pq = 1.0 + p*p + q*q
            pqs = 1.0 + ps*ps + qs*qs

            # -- compute the reflectance map
            self.reflectance_map = (1.0 + p*ps + q*qs) / (np.sqrt(pq) * np.sqrt(pqs))
            self.reflectance_map = np.maximum(np.zeros(image_shape), self.reflectance_map)

            fz = -1.0 * ((self.image/max_pixel) - self.reflectance_map)
            dfz = -1.0 * ((ps+qs)/(np.sqrt(pq)*np.sqrt(pqs))-((p+q)*(1.0+p*ps+q*qs)/(np.sqrt(pq*pq*pq)*np.sqrt(pqs))))

            y = fz + (dfz * zk1)
            k = (sk1 * dfz) / (self.Wn + sk1 * dfz * dfz)

            sk = (1.0 - (k * dfz)) * sk1
            self.depth_map = zk1 + k * (y - (dfz * zk1))

            # -- update depth map
            zk1 = self.depth_map
            sk1 = sk

            # -- estimation of the light direction for next iteration

            l_dot_n = np.abs(np.dot(self.normals, np.array(light)))
            l_dot_n[l_dot_n == 0.] = 1.

            self.albedo_map = self.image / l_dot_n
            self.albedo_map[self.albedo_map > 255] = 255.
            self.albedo_map[self.albedo_map < 1.] = 1.

        self.reflectance_map = self.reflectance_map / np.pi

    def __compute_local_sfs(self):

        n_rows, n_cols = self.image.shape[:2]

        self.depth_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        self.reflectance_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        self.albedo_map = np.zeros((n_rows, n_cols), dtype=np.float32)

        sk = np.zeros((n_rows, n_cols), dtype=np.float32)
        fzk = np.zeros((n_rows, n_cols), dtype=np.float32)
        dfzk = np.zeros((n_rows, n_cols), dtype=np.float32)

        zk1 = np.zeros((n_rows, n_cols), dtype=np.float32)
        sk1 = np.ones((n_rows, n_cols), dtype=np.float32)

        ps = np.zeros((n_rows, n_cols), dtype=np.float32)
        qs = np.ones((n_rows, n_cols), dtype=np.float32)

        for it in range(self.NB_ITERATIONS):

            self.normals = []
            for i in range(1, n_rows):
                for j in range(1, n_cols):

                    # -- compute dz/dx
                    if (j - 1) >= 0:
                        p = zk1[i, j] - zk1[i, (j - 1)]
                    else:
                        p = 0.0

                    # -- compute dz/dy
                    if (i - 1) >= 0:
                        q = zk1[i, j] - zk1[(i - 1), j]
                    else:
                        q = 0.0

                    n = np.array([-1.*p, -1.*q, 1.])
                    n /= np.linalg.norm(n)
                    self.normals += [n]

                    # -- compute the reflectance map
                    pq = 1.0 + p * p + q * q
                    pqs = 1.0 + ps[i, j] * ps[i, j] + qs[i, j] * qs[i, j]
                    rij = max(0.0, (1 + p * ps[i, j] + q * qs[i, j]) / (math.sqrt(pq) * math.sqrt(pqs)))
                    self.reflectance_map[i, j] = rij / math.pi

                    # -- compute fz function and its derivate
                    eij = self.image[i, j] / self.image.max()

                    fz = -1.0 * (eij - rij)
                    dfz = -1.0 * ((ps[i, j] + qs[i, j]) / (math.sqrt(pq) * math.sqrt(pqs)) - (p + q) *
                                  (1.0 + p * ps[i, j] + q * qs[i, j]) / (math.sqrt(pq * pq * pq) * math.sqrt(pqs)))

                    y = fz + (dfz * zk1[i, j])
                    k = (sk1[i, j] * dfz) / (self.Wn + sk1[i, j] * dfz * dfz)

                    sk[i, j] = (1.0 - (k * dfz)) * sk1[i, j]
                    self.depth_map[i, j] = zk1[i, j] + k * (y - (dfz * zk1[i, j]))

                    fzk[i, j] = fz
                    dfzk[i, j] = dfz

            # -- update depth map
            for i in range(n_rows):
                for j in range(n_cols):
                    zk1[i, j] = self.depth_map[i, j]
                    sk1[i, j] = sk[i, j]

            # -- estimation of the light direction
            xl, yl, zl = 0.01, 0.01, 1.00

            self.normals = np.array(self.normals)
            self.normals = np.reshape(self.normals, (n_rows - 1, n_cols - 1, 3))
            zero_row = np.zeros((1, n_cols - 1, 3), dtype=np.float32)
            zero_col = np.zeros((n_rows, 1, 3), dtype=np.float32)

            self.normals = np.concatenate((zero_row, self.normals), axis=0)
            self.normals = np.concatenate((zero_col, self.normals), axis=1)

            for y in range(n_rows):
                for x in range(n_cols):

                    # if y != 0 and x != 0:
                    if np.abs(np.dot(self.normals[y, x, :], np.array([xl, yl, zl]))) != 0:
                        self.albedo_map[y, x] = self.image[y, x] / np.abs(np.dot(self.normals[y, x, :],
                                                                                 np.array([xl, yl, zl])))

                    else:
                        self.albedo_map[y, x] = self.image[y, x]

            self.albedo_map[self.albedo_map > 255] = 255.
            self.albedo_map[self.albedo_map < 1.] = 1.

    def compute_sfs(self):

        #if self.image is None:
        self.load_image()

        if self.local_estimation:
            self.__compute_local_sfs()
        else:
            self.__compute_global_sfs()

        self.depth_map = self.normalization(self.depth_map)
        self.reflectance_map = self.normalization(self.reflectance_map)

        self.albedo_free_image = self.image/self.albedo_map
        self.albedo_free_depth = self.depth_map/self.albedo_map

        self.albedo_free_image = self.normalization(self.albedo_free_image)
        self.albedo_free_depth = self.normalization(self.albedo_free_depth)

        self.albedo_map = np.round(self.albedo_map).astype(np.uint8)
        self.depth_map = np.round(self.depth_map).astype(np.uint8)
        self.reflectance_map = np.round(self.reflectance_map).astype(np.uint8)
        self.albedo_free_image = np.round(self.albedo_free_image).astype(np.uint8)
        self.albedo_free_depth = np.round(self.albedo_free_depth).astype(np.uint8)

        return self.depth_map,self.reflectance_map,self.albedo_map


    def show_depth_image(self):

        plot_col = 4
        plot_row = 2
        fs = 10

        fig = plt.figure(figsize=(8, 6), dpi=100)

        titles = ['Original', 'Reflectance Map', 'Depth Map', 'Albedo Map', 'Nx', 'Ny', 'Nz', '']
        data_plt = [self.image, self.reflectance_map, self.depth_map, self.albedo_map,
                    self.albedo_free_image, self.albedo_free_depth, self.normals[:, :, 2],
                    np.ones(self.image.shape, dtype=np.uint8) * 255,
                    np.ones(self.image.shape, dtype=np.uint8) * 255,
                    ]

        axis = []
        for k in range((plot_col * plot_row)):
            axis += [fig.add_subplot(plot_row, plot_col, k + 1)]

        for k in range((plot_col * plot_row)):
            axis[k].imshow(data_plt[k])
            axis[k].set_title(titles[k], fontsize=fs)
            axis[k].set_xticklabels([])
            axis[k].set_yticklabels([])

        plt.show()

    @staticmethod
    def show_normal_surface():

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.8))

        u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
        v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
        w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
             np.sin(np.pi * z))

        ax.quiver(x, y, z, u, v, w, length=0.1)

        plt.show()


def Anderson(image_path, local_estimation=False, light_direction='constant'):
    #pdb.set_trace()
    image_input = np.asarray(cv2.resize(cv2.imread(image_path, -1),(150,150),interpolation=cv2.INTER_AREA))

    sfs0 = Tsai(input_fname=image_input[:,:,0], local_estimation=local_estimation, light_direction=light_direction)
    depth_map0, reflectance_map0,albedo_map0 = sfs0.compute_sfs()
    sfs1 = Tsai(input_fname=image_input[:, :, 1], local_estimation=local_estimation, light_direction=light_direction)
    depth_map1, reflectance_map1, albedo_map1 = sfs1.compute_sfs()
    sfs2 = Tsai(input_fname=image_input[:, :, 2], local_estimation=local_estimation, light_direction=light_direction)
    depth_map2, reflectance_map2, albedo_map2 = sfs2.compute_sfs()

    depth_map = np.concatenate((np.expand_dims(depth_map0,axis=2),np.expand_dims(depth_map1,axis=2),np.expand_dims(depth_map2,axis=2)), axis=2)
    reflectance_map = np.concatenate(
        (np.expand_dims(reflectance_map0, axis=2), np.expand_dims(reflectance_map1, axis=2), np.expand_dims(reflectance_map2, axis=2)),
        axis=2)
    albedo_map = np.concatenate(
        (np.expand_dims(albedo_map0, axis=2), np.expand_dims(albedo_map1, axis=2), np.expand_dims(albedo_map2, axis=2)),
        axis=2)
    del sfs0, sfs1, sfs2, depth_map0, depth_map1, depth_map2, reflectance_map0, reflectance_map1, reflectance_map2, albedo_map0, albedo_map1, albedo_map2
    return image_input,depth_map,reflectance_map,albedo_map

def extract_dra(image_input: object, output_size: object = (256, 256), local_estimation: object = False, light_direction: object = 'constant') -> object:
    #pdb.set_trace()
    image_input = np.asarray(cv2.resize(image_input,output_size,interpolation=cv2.INTER_AREA))

    sfs0 = Tsai(input_fname=image_input[:,:,0], local_estimation=local_estimation, light_direction=light_direction)
    depth_map0, reflectance_map0,albedo_map0 = sfs0.compute_sfs()
    sfs1 = Tsai(input_fname=image_input[:, :, 1], local_estimation=local_estimation, light_direction=light_direction)
    depth_map1, reflectance_map1, albedo_map1 = sfs1.compute_sfs()
    sfs2 = Tsai(input_fname=image_input[:, :, 2], local_estimation=local_estimation, light_direction=light_direction)
    depth_map2, reflectance_map2, albedo_map2 = sfs2.compute_sfs()

    depth_map = np.concatenate((np.expand_dims(depth_map0,axis=2),np.expand_dims(depth_map1,axis=2),np.expand_dims(depth_map2,axis=2)), axis=2)
    reflectance_map = np.concatenate(
        (np.expand_dims(reflectance_map0, axis=2), np.expand_dims(reflectance_map1, axis=2), np.expand_dims(reflectance_map2, axis=2)),
        axis=2)
    albedo_map = np.concatenate(
        (np.expand_dims(albedo_map0, axis=2), np.expand_dims(albedo_map1, axis=2), np.expand_dims(albedo_map2, axis=2)),
        axis=2)
    del sfs0, sfs1, sfs2, depth_map0, depth_map1, depth_map2, reflectance_map0, reflectance_map1, reflectance_map2, albedo_map0, albedo_map1, albedo_map2
    return image_input,depth_map,reflectance_map,albedo_map

if __name__ == '__main__':
    original_image, Depth_map, Reflectance_map, Albedo_map = Anderson(r'1.jfif')
    cv2.imwrite('./albedo1.jpg', Albedo_map, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite('./Depth_map.jpg', Depth_map, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite('./Reflectance_map.jpg', Reflectance_map, [int( cv2.IMWRITE_JPEG_QUALITY), 100])