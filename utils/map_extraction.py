"""
References:
1. ColorLBP computation referenced by:
   - https://github.com/KevinDepedri/Face-Spoofing-Detection-Using-Colour-Texture-Analysis/

2. Albedo, Reflectance, and Depth computation referenced by:
   - https://github.com/allansp84/shape-from-shading-for-face-pad
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class MapGenerator:
    """
    A class for generating Local Binary Pattern (LBP) maps and performing Shape-from-Shading (SFS) computations.

    Attributes:
        image (np.ndarray): Grayscale version of the input image used for SFS computations.
        original_image (np.ndarray): Original RGB/BGR version of the input image.
        input_fname (str): Path to the input image file.
        output_fname (str): Name of the output file.
        depth_map (np.ndarray): Computed depth map from SFS.
        albedo_map (np.ndarray): Computed albedo map from SFS.
        reflectance_map (np.ndarray): Computed reflectance map from SFS.
        normals (np.ndarray): Normal vectors computed during SFS.
        color_lbp_map (np.ndarray): ColorLBP map derived from RGB channels.
        NB_ITERATIONS (int): Number of iterations for the SFS algorithm.
        Wn (float): Regularization parameter for the SFS algorithm.
        light_direction (str): Method to estimate light direction (default: 'constant').
    """
    def __init__(self, image=None, input_fname='', output_fname='', light_direction='constant'):
        self.image = image
        self.original_image = None  # 원본 RGB/BGR 이미지를 저장
        self.input_fname = input_fname
        self.output_fname = output_fname
        self.depth_map = None
        self.albedo_map = None
        self.reflectance_map = None
        self.normals = None
        self.albedo_free_image = None
        self.albedo_free_depth = None
        self.NB_ITERATIONS = 5
        self.Wn = 0.001
        self.light_direction = light_direction
        self.color_lbp_map = None

    def load_image(self):
        """
        Initialize the MapGenerator class with input and output filenames, and light direction estimation.

        Args:
            image (np.ndarray, optional): Grayscale image for SFS computations. Defaults to None.
            input_fname (str): Path to the input image file.
            output_fname (str): Name of the output file.
            light_direction (str): Method for light direction estimation. Defaults to 'constant'.
        """
        self.original_image = cv2.imread(self.input_fname)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from {self.input_fname}")

        self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        print("[INFO] Image successfully loaded.")
        # print(f"Original image shape (RGB/BGR): {self.original_image.shape}")
        # print(f"Grayscale image shape: {self.image.shape}")


    @staticmethod
    def normalization(data, min_value=0., max_value=255.):
        """
        Normalize data to a specified range [min_value, max_value].

        Formula:
            normalized = ((data - data_min) / (data_max - data_min)) * (max_value - min_value) + min_value

        Args:
            data (np.ndarray): Input data to normalize.
            min_value (float): Minimum value of the normalized range.
            max_value (float): Maximum value of the normalized range.

        Returns:
            np.ndarray: Normalized data.
        """
        data = np.array(data, dtype=np.float32)
        data_min = np.min(data)
        data_max = np.max(data)

        if data_min != data_max:
            data_norm = (data - data_min) / (data_max - data_min)
            data_scaled = data_norm * (max_value - min_value) + min_value
        else:
            data_scaled = data

        return data_scaled

    # @staticmethod
    # def compute_lbp(img, radius=1, neighbors=8):
    #     """
    #     Compute Uniform Local Binary Pattern (LBP) for a single channel.

    #     The LBP is calculated by comparing the center pixel value with its neighbors in a circular pattern.
    #     The method also checks for uniformity by counting transitions in the binary pattern.

    #     Args:
    #         img (np.ndarray): Input grayscale image.
    #         radius (int): Radius of the circular neighborhood.
    #         neighbors (int): Number of neighbors in the circular pattern.

    #     Returns:
    #         np.ndarray: Uniform LBP map of the input image.
    #     """
    #     height, width = img.shape
    #     lbp = np.zeros((height, width), dtype=np.uint8)

    #     # Precompute relative positions of neighbors based on radius and neighbors
    #     angles = [2.0 * np.pi * n / neighbors for n in range(neighbors)]
    #     offsets = [(int(round(radius * np.sin(a))), int(round(radius * np.cos(a)))) for a in angles]

    #     for x in range(radius, height - radius):
    #         for y in range(radius, width - radius):
    #             center = img[x, y]
    #             binary_pattern = []

    #             # Collect binary pattern by comparing neighbors with the center
    #             for dy, dx in offsets:
    #                 neighbor = img[x + dy, y + dx]
    #                 binary_pattern.append(1 if neighbor >= center else 0)

    #             # Count transitions in the binary pattern
    #             transitions = sum(abs(binary_pattern[i] - binary_pattern[i - 1]) for i in range(neighbors))

    #             # Determine if the pattern is uniform
    #             if transitions <= 2:
    #                 # Compute uniform LBP value
    #                 lbp_value = sum(binary_pattern[i] * (1 << i) for i in range(neighbors))
    #             else:
    #                 # Assign a fixed value for non-uniform patterns
    #                 lbp_value = neighbors * (neighbors - 1) + 2

    #             lbp[x, y] = lbp_value

    #     return lbp
    # def compute_color_lbp(self, radius=1, neighbors=8):
    #     """
    #     Compute Uniform ColorLBP by calculating Uniform LBP maps for each RGB channel.

    #     Each channel's LBP map is computed independently using Uniform LBP logic, and the final result
    #     is a merged map of the three channels.

    #     Args:
    #         radius (int): Radius of the circular neighborhood for LBP.
    #         neighbors (int): Number of neighbors in the circular pattern.

    #     Raises:
    #         ValueError: If `self.original_image` is not loaded.
    #     """
    #     if self.original_image is None:
    #         raise ValueError("Original RGB/BGR image not loaded. Ensure `load_image` is called before this method.")

    #     # Split BGR channels
    #     b, g, r = cv2.split(self.original_image)

    #     # Compute Uniform LBP for each channel
    #     lbp_r = self.compute_lbp(r, radius, neighbors)
    #     lbp_g = self.compute_lbp(g, radius, neighbors)
    #     lbp_b = self.compute_lbp(b, radius, neighbors)

    #     # Merge LBP results into a Uniform ColorLBP map
    #     self.color_lbp_map = cv2.merge([lbp_b, lbp_g, lbp_r])  
    
    @staticmethod
    def compute_lbp(img, radius=1, neighbors=8):
        """
        Compute Local Binary Pattern (LBP) for a single channel.

        The LBP of a pixel is computed by comparing the center pixel value with its neighbors
        in a circular pattern. A binary code is generated based on whether each neighbor is greater
        than or equal to the center pixel.

        Formula:
            binary_code[i] = 1 if neighbor[i] >= center else 0
            lbp_value = sum(binary_code[i] * (2 ** i)) for i in range(neighbors)

        Args:
            img (np.ndarray): Input grayscale image.
            radius (int): Radius of the circular neighborhood.
            neighbors (int): Number of neighbors in the circular pattern.

        Returns:
            np.ndarray: LBP map of the input image.
        """
        height, width = img.shape
        lbp = np.zeros((height, width), dtype=np.uint8)

        for x in range(radius, height - radius):
            for y in range(radius, width - radius):
                center = img[x, y]
                binary = []
                for n in range(neighbors):
                    theta = 2.0 * np.pi * n / neighbors
                    dx = int(round(radius * np.cos(theta)))
                    dy = int(round(radius * np.sin(theta)))
                    neighbor = img[x + dx, y + dy]
                    binary.append(1 if neighbor >= center else 0)
                lbp[x, y] = sum([val * (1 << idx) for idx, val in enumerate(binary)])

        return lbp

    def compute_color_lbp(self, radius=1, neighbors=8):
        """
        Compute ColorLBP by calculating LBP maps for each RGB channel.

        Each channel's LBP map is computed independently, and the final result
        is a merged map of the three channels.

        Args:
            radius (int): Radius of the circular neighborhood for LBP.
            neighbors (int): Number of neighbors in the circular pattern.

        Raises:
            ValueError: If `self.original_image` is not loaded.
        """
        if self.original_image is None:
            raise ValueError("Original RGB/BGR image not loaded. Ensure `load_image` is called before this method.")

        # Split BGR channels
        b, g, r = cv2.split(self.original_image)

        # Compute LBP for each channel
        lbp_r = self.compute_lbp(r, radius, neighbors)
        lbp_g = self.compute_lbp(g, radius, neighbors)
        lbp_b = self.compute_lbp(b, radius, neighbors)

        # Merge LBP results into a ColorLBP map
        self.color_lbp_map = cv2.merge([lbp_b, lbp_g, lbp_r])
  

    def __compute_global_sfs(self):
        """
        Perform global Shape-from-Shading (SFS) computation.

        The algorithm estimates depth, reflectance, and albedo maps iteratively
        based on the image intensity and light direction.

        Formulas:
            Reflectance Map:
                R(p, q) = (1 + p * ps + q * qs) / (sqrt(1 + p^2 + q^2) * sqrt(1 + ps^2 + qs^2))

            Depth Map:
                fz = -1 * ((I / max_pixel) - R)
                dfz = derivative of fz w.r.t depth
                depth_map = zk1 + k * (fz - dfz * zk1)

            Albedo Map:
                A(x, y) = I(x, y) / max(1, |N(x, y) · L|)

        Args:
            None
        """
        self.image = np.array(self.image, dtype=np.float32)
        image_shape = self.image.shape

        # Initialization
        self.depth_map = np.zeros(image_shape, dtype=np.float32)
        self.reflectance_map = np.zeros(image_shape, dtype=np.float32)
        self.albedo_map = np.zeros(image_shape, dtype=np.float32)

        zk1 = np.zeros(image_shape, dtype=np.float32)
        sk1 = np.ones(image_shape, dtype=np.float32)

        # Light direction initialization
        if 'constant' in self.light_direction:
            light = [0.1, 0.1, 1.0]
            xl, yl, zl = light
        else:
            raise Exception('Method for light estimation not found!')

        max_pixel = self.image.max()
        if max_pixel == 0:
            raise ValueError("Invalid max pixel value in the image!")

        ps = np.ones(image_shape, dtype=np.float32) * (xl / zl)
        qs = np.ones(image_shape, dtype=np.float32) * (yl / zl)

        # Iterative computation
        for _ in range(self.NB_ITERATIONS):
            # Gradient computation
            p, q = np.gradient(zk1)

            # Normals computation
            self.normals = np.dstack([-p, -q, np.ones(p.shape)])
            norm = np.linalg.norm(self.normals, axis=2)
            self.normals[:, :, 0] /= norm
            self.normals[:, :, 1] /= norm
            self.normals[:, :, 2] /= norm

            # Reflectance map computation
            pq = 1.0 + p * p + q * q
            pqs = 1.0 + ps * ps + qs * qs
            self.reflectance_map = (1.0 + p * ps + q * qs) / (np.sqrt(pq) * np.sqrt(pqs))
            self.reflectance_map = np.maximum(0.0, self.reflectance_map)

            # Depth map computation
            fz = -1.0 * ((self.image / max_pixel) - self.reflectance_map)
            dfz = -1.0 * ((ps + qs) / (np.sqrt(pq) * np.sqrt(pqs)) -
                          ((p + q) * (1.0 + p * ps + q * qs) /
                           (np.sqrt(pq * pq * pq) * np.sqrt(pqs))))

            y = fz + (dfz * zk1)
            k = (sk1 * dfz) / (self.Wn + sk1 * dfz * dfz)

            sk = (1.0 - (k * dfz)) * sk1
            self.depth_map = zk1 + k * (y - (dfz * zk1))

            # Update zk1 and sk1
            zk1 = self.depth_map
            sk1 = sk

            # Albedo map computation
            l_dot_n = np.abs(np.dot(self.normals, np.array(light)))
            l_dot_n[l_dot_n == 0.] = 1.  # Prevent division by zero
            self.albedo_map = self.image / l_dot_n
            self.albedo_map[self.albedo_map > 255] = 255.
            self.albedo_map[self.albedo_map < 1.] = 1.

        # Normalize reflectance map
        self.reflectance_map = self.reflectance_map / np.pi


    def compute_sfs(self):
        """
        Compute Shape-from-Shading (SFS) to estimate depth, reflectance, and albedo maps.

        This method prepares the grayscale image, performs global SFS computations,
        and normalizes the results.
        """
        if self.image is None:
            self.load_image()

        self.__compute_global_sfs()

        # Normalize results
        self.depth_map = self.normalization(self.depth_map)
        self.reflectance_map = self.normalization(self.reflectance_map)

    def save_results(self, output_dir, save_img=True):
        """
        Save results (original, depth, reflectance, albedo, and ColorLBP maps) to the specified directory.

        If `save_img` is True, results are saved as PNG images. Otherwise, they are saved as .npy files.

        Args:
            output_dir (str): Path to the output directory.
            save_img (bool): Whether to save results as images (default: True).
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.output_fname))[0] if self.output_fname else 'output'

        save_list = [
            ('original', self.original_image),  # Save original RGB/BGR image
            ('depth', self.depth_map),
            ('reflectance', self.reflectance_map),
            ('albedo', self.albedo_map),
            ('color_lbp', self.color_lbp_map)
        ]

        for name, data in save_list:
            if data is None:
                continue
            
            output_path = os.path.join(output_dir, f"{base_name}_{name}.png" if save_img else f"{base_name}_{name}.npy")
            
            if save_img:
                if name == 'original':
                    # Save original image as is
                    cv2.imwrite(output_path, data)
                elif name in ['depth', 'reflectance', 'albedo']:
                    # Normalize and apply color mapping
                    norm_data = np.round(self.normalization(data)).astype(np.uint8)
                    color_mapped = cv2.applyColorMap(norm_data, cv2.COLORMAP_JET)

                    # Blend with the original image
                    blended = cv2.addWeighted(self.original_image, 0.7, color_mapped, 0.2, 0.1)

                    # Save the blended image
                    cv2.imwrite(output_path, blended)
                else:
                    # Save as grayscale or raw RGB data
                    if len(data.shape) == 3 and data.shape[2] == 3:  # RGB/BGR image
                        cv2.imwrite(output_path, data)
                    else:  # Grayscale
                        norm_data = np.round(self.normalization(data)).astype(np.uint8)
                        cv2.imwrite(output_path, norm_data)
            else:
                # Save as .npy file for raw data
                np.save(output_path, data)

def main(args):
    # Initialize the Tsai object
    map_gen = MapGenerator(input_fname=args.input_path, output_fname="output_image")

    # Ensure the image is loaded before any processing
    print("[INFO] Loading image...")
    map_gen.load_image()

    # Perform the requested operations
    print("[INFO] Computing ColorLBP...")
    map_gen.compute_color_lbp()
    print("[INFO] Computing Shape-from-Shading...")
    map_gen.compute_sfs()

    # Save results
    print("[INFO] Saving results...")
    map_gen.save_results(args.output_path, save_img=args.save_img)
    print("[INFO] Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified ColorLBP and TSai module.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--save_img", action="store_true", help="Save output as images. Otherwise, saves as .npy files.")
    args = parser.parse_args()

    main(args)

# python map_extractor.py --input_path crop_0000.jpg --output_path ./output --save_img 