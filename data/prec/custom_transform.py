import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image
class ShufflePatch(torch.nn.Module):
    """
        Import
    """
    def __init__(self, patch_size=32, shuffle_prob=0.5, config=None):

        super(ShufflePatch, self).__init__()
        # im is of the size: [H, W, C]
        self.patch_size = patch_size
        self.shuffle_prob = shuffle_prob

    def shuffle_patches(self, im):
        img_patches = []
        img_patches
        H, W, C = im.shape

        num_patch_h = H // self.patch_size
        num_patch_w = W // self.patch_size

        indices_w = np.linspace(0, W, num_patch_w + 1, endpoint=True, dtype=int)
        indices_h = np.linspace(0, H, num_patch_h + 1, endpoint=True, dtype=int)

        patches = []
        for i in range(num_patch_w):
            for j in range(num_patch_h):
                start_w, end_w = indices_w[i], indices_w[i + 1]
                start_h, end_h = indices_h[i], indices_h[i + 1]
                patch = im[start_h:end_h, start_w:end_w, :]
                patches.append(patch)
        random.shuffle(patches)
        new_im = np.zeros_like(im)
        for i in range(num_patch_w):
            for j in range(num_patch_h):
                start_w, end_w = indices_w[i], indices_w[i + 1]
                start_h, end_h = indices_h[i], indices_h[i + 1]
                new_im[start_h:end_h, start_w:end_w, :] = patches[i*num_patch_w+j]

        return new_im

    def forward(self, im):
        p = np.random.uniform()


        if p < self.shuffle_prob:
            if 'PIL' in str(type(im)):
                # PIL to im narray
                im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
                im = self.shuffle_patches(im)

                return Image.fromarray(im.astype(np.uint8))
        else:
            return im


if __name__ == '__main__':
    x =  np.zeros([256,256,3])
    trans = ShufflePatch()
    xx = trans(x)
    import IPython; IPython.embed()

